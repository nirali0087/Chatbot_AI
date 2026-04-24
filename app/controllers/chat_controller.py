from flask import Blueprint, request, render_template, current_app, jsonify, redirect, url_for
from flask_login import current_user, login_required
from app.models import db, Conversation, Message
from app.services.vector_utils import (
    get_embedding, summarize_history, find_similar_answer, calculate_context_similarity,
    should_use_web_fallback, enhanced_prompt_template, search_user_messages, enhance_with_mcp, self_repair_llm_response, summarize_text,
    summarize_text
)
import datetime
from app.services.web_search import web_search_fallback, format_web_results

chat_bp = Blueprint('chat', __name__, template_folder='../views/templates/chat')

############################################
# Unified helper for handling chat requests
############################################

def _truncate_title(text: str, max_len: int = 30) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text



@chat_bp.route('/', methods=['GET', 'POST'])
@login_required
def home():
    conversations = Conversation.query.filter_by(user_id=current_user.id).order_by(Conversation.created_at.desc()).all()
    conversation_messages = []
    active_conv_id = None

    if request.method == 'POST':
        # Always create a new conversation and redirect to its page with the question
        question = request.form.get('question')
        use_context = request.form.get('use_context', 'true').lower() == 'true'

        try:
            title = _truncate_title(question) if question else "New Conversation"
            conversation = Conversation(user_id=current_user.id, title=title)
            db.session.add(conversation)
            db.session.commit()

            return redirect(url_for('chat.load_conversation', conversation_id=conversation.id, q=question or '', use_context=str(use_context).lower()))

        except Exception as e:
            db.session.rollback()
            return render_template('chat/index.html', 
                                 conversations=conversations,
                                 conversation_messages=conversation_messages,
                                 active_conversation=active_conv_id,
                                 error=f"Error: {str(e)}")

    return render_template('chat/index.html', 
                         conversations=conversations,
                         conversation_messages=conversation_messages,
                         active_conversation=active_conv_id)

@chat_bp.route('/conversation/<int:conversation_id>')
@login_required
def get_conversation(conversation_id):
    conversation = Conversation.query.filter_by(id=conversation_id, user_id=current_user.id).first_or_404()
    messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp.asc()).all()
    
    return jsonify({
        'id': conversation.id,
        'title': conversation.title,
        'messages': [{
            'id': msg.id,
            'is_user': msg.is_user,
            'content': msg.content,
            'timestamp': msg.timestamp.isoformat()
        } for msg in messages]
    })

@chat_bp.route('/conversation/<int:conversation_id>/delete', methods=['POST'])
@login_required
def delete_conversation(conversation_id):
    conversation = Conversation.query.filter_by(id=conversation_id, user_id=current_user.id).first_or_404()
    db.session.delete(conversation)
    db.session.commit()
    return redirect(url_for('chat.home'))

@chat_bp.route('/load_conversation/<int:conversation_id>', methods=['GET', 'POST'])
@login_required
def load_conversation(conversation_id):
    conversation = Conversation.query.filter_by(id=conversation_id, user_id=current_user.id).first_or_404()
    messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp.asc()).all()
    conversations = Conversation.query.filter_by(user_id=current_user.id).order_by(Conversation.created_at.desc()).all()

    if request.method == 'POST' or request.args.get('q'):
        question = request.form.get('question') or request.args.get('q')
        use_context = request.form.get('use_context', 'true').lower() == 'true'
        
        vector_store = current_app.vector_store
        llm = current_app.llm

        try:
            user_message = Message(
                conversation_id=conversation.id,
                is_user=True,
                content=question,
                timestamp=datetime.datetime.utcnow()
            )

            question_embedding = None
            if use_context:
                question_embedding = get_embedding(question)
                if question_embedding is not None:
                    user_message.set_embedding(question_embedding)

            db.session.add(user_message)
            db.session.flush()

            # Build context if needed
            relevant_contexts = []
            used_web_search = False
            web_context = None
            if use_context:
                # Similar past Q/A
                similar_answer_data = None
                if question_embedding is not None:
                    similar_answer_data = find_similar_answer(current_user.id, question_embedding, top_k=3, conversation_id=conversation.id)

                context_parts = []
                if similar_answer_data:
                    print("using similar past Q & A for context.")
                    past_qa_context = f"""
                    A similar past conversation might help:
                    User asked: "{similar_answer_data['user_message'].content}"
                    AI answered: "{similar_answer_data['ai_answer'].content}"
                    """
                    context_parts.append(past_qa_context.strip())
                else:
                    print("No similar past Q&A found.")

                # Similar user messages
                user_similar_msgs = []
                if question_embedding is not None:
                    all_similar_msgs = search_user_messages(user_id=current_user.id, query_embedding=question_embedding, top_k=5, conversation_id=conversation.id)
                    
                    normalized_question = (question or "").strip().lower()
                    user_similar_msgs = [
                        m for m in all_similar_msgs
                        if m.content.strip().lower() != normalized_question
                    ]
                
                    if user_similar_msgs:
                        print(f"Found {len(user_similar_msgs)} similar user messages for context.....")
                        user_message_context = "\n\n".join([
                            f"{'user' if m.is_user else 'assistant'} : {m.content}"
                            for m in user_similar_msgs
                        ])
                        context_parts.append(user_message_context)
                    else:
                        print("No similar user messages found...")


                # External docs from FAISS
                if vector_store is not None:
                    relevant_docs = vector_store.similarity_search(question, k=4)
                    if relevant_docs:
                        print(f"DEBUG: Retrieved {len(relevant_docs)} documents from FAISS doc store.")
                        relevant_contexts = [summarize_text(doc.page_content, max_sentences=5, max_chars=1000) for doc in relevant_docs]
                        context_parts.append("\n\n".join([rc for rc in relevant_contexts if rc]))
                    else:
                        print("DEBUG: No relevant documents found in FAISS doc store.")
                else:
                    print("Vector store not available.")


                document_context = "\n\n".join([p for p in context_parts if p])

                # Web fallback decision
                context_similarity_score = calculate_context_similarity(question, document_context)
                has_context = bool((relevant_contexts or []) + (user_similar_msgs or []))
                used_web_search = should_use_web_fallback(
                    question,
                    context_similarity_score,
                    has_context
                )
                print(f"Context similarity score: {context_similarity_score}")
                print(f"Should use web fallback: {used_web_search}")

                if used_web_search:
                    print("Using web search fallback.....")
                    web_results = web_search_fallback(question)
                    web_context = format_web_results(web_results)
                else:
                    print("Web search fallback not used.")


                # Build recent conversation history (across user's conversations)
                recent_messages = (
                    Message.query
                    .filter(Message.conversation_id == conversation.id)
                    .order_by(Message.timestamp.desc())
                    .limit(20)
                    .all()
                )
                recent_messages.reverse()
                conversation_history = summarize_history(recent_messages, llm=current_app.llm)

                prompt =(enhanced_prompt_template(
                        history=conversation_history,
                        context=document_context,
                        user_input=question,
                        web_context=web_context
                    ) if use_context else question )

                initial_response = llm.invoke(prompt)
                answer_text = initial_response.content
                    
                if question_embedding is not None:
                    if use_context:
                        repaired_response, was_repaired = self_repair_llm_response(
                            llm=llm,
                            question=question,
                            original_context=document_context,
                            web_context=web_context,
                            conversation_messages=recent_messages,
                            question_embedding=question_embedding,
                            initial_response=answer_text
                        )
                        
                        if was_repaired:
                            answer_text = repaired_response
                            print("Used MCP self-repair for improved response")
                        else:
                            print("Used original LLM response")
                    
                    else:
                    # Direct, no context - still use MCP for basic analysis
                        print("Context not used. Sending direct question to LLM.")
                        mcp_result = enhance_with_mcp(
                            question=question,
                            context="",
                            conversation_messages=recent_messages,
                            question_embedding=question_embedding,
                            initial_response=answer_text
                        )
                        
                        if mcp_result["needs_enhancement"] and mcp_result["confidence_score"] < 0.6:
                            # If confidence is low even without context, try a refined approach
                            refined_question = f"Please answer this question clearly: {question}"
                            refined_response = llm.invoke(refined_question)
                            answer_text = refined_response.content
                            print("Used MCP-refined question approach")
                

            # Save AI response
            ai_message = Message(
                conversation_id=conversation.id,
                is_user=False,
                content=answer_text,
                timestamp=datetime.datetime.utcnow()
            )
            if use_context:
                answer_embedding = get_embedding(answer_text)
                if answer_embedding is not None:
                    ai_message.set_embedding(answer_embedding)
            db.session.add(ai_message)
            db.session.commit()

            if not conversation.title:
                conversation.title = _truncate_title(question)
                db.session.commit()

            messages = Message.query.filter_by(conversation_id=conversation.id).order_by(Message.timestamp.asc()).all()

            return render_template(
                'chat/index.html',
                conversations=conversations,
                active_conversation=conversation.id,
                conversation_messages=messages,
                question=question,
                answer=answer_text,
                relevant_contexts=relevant_contexts,
                is_repeated_question=False,
                use_context=use_context,
                used_web_search=used_web_search if use_context else False
            )

        except Exception as e:
            db.session.rollback()
            return render_template('chat/index.html', 
                                 conversations=conversations,
                                  conversation_messages=messages,
                                  active_conversation=conversation.id,
                                  error=f"Error: {str(e)}")
    
    return render_template(
        'chat/index.html',
        conversations=conversations,
        active_conversation=conversation.id,
        conversation_messages=messages,
        relevant_contexts=[]
    )
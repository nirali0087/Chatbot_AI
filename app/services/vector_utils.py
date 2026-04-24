from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from app.models import Message, Conversation
import re
import requests
from bs4 import BeautifulSoup
from googlesearch import search as google_search
import time
import json
from urllib.parse import quote_plus
from app.config import Config
from langchain_ollama import OllamaEmbeddings, ChatOllama
from sqlalchemy import text
from app import db
from app.models import Message, Conversation
import numpy as np
from app.services.mcp_client import mcp_client


embeddings_model = OllamaEmbeddings(model=Config.OLLAMA_EMBEDDING_MODEL, base_url=Config.OLLAMA_EMBEDDING_URL)
SIMILARITY_THRESHOLD = 0.8  

def initialize_vector_store():
    try:
        vector_store = FAISS.load_local(
            Config.FAISS_INDEX_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        print("FAISS index loaded successfully")
        return vector_store
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def initialize_llm():
    try:
        llm = ChatOpenAI(
            openai_api_base=Config.LM_STUDIO_BASE_URL,
            openai_api_key="not-needed",
            model=Config.LM_STUDIO_LLM_MODEL,
            # temperature=0.1
        )

        # llm = ChatOpenAI(
        #     openai_api_key="your_openai_api_key_here",
        #     model="o4-mini"
        # )
        print("LM Studio LLM initialized successfully")
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

def get_embedding(text):
    """Generate embedding for text using nomic-embed-text model"""
    try:
        if not text or not text.strip():
            return None
        
        text = clean_text(text)
        return embeddings_model.embed_query(text)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None



def clean_text(text):
    """Clean text for better embedding generation"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    return text

def summarize_text(text: str, max_sentences: int = 5, max_chars: int = 1200) -> str:
    """Simple heuristic summarizer: take up to `max_sentences` sentences and trim to max_chars."""
    if not text:
        return ""
    # Break into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    selected = sentences[:max_sentences]
    joined = ' '.join(selected)
    # Trim to max chars without cutting mid-word
    if len(joined) <= max_chars:
        return joined
    return joined[:max_chars].rsplit(' ', 1)[0] + '...'


def limit_text(text: str, max_chars: int = 1500) -> str:
    """Hard trim text to max_chars (preserving words)."""
    if not text:
        return ""
    t = re.sub(r'\s+', ' ', text).strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rsplit(' ', 1)[0] + '...'


def search_user_messages(user_id, query_embedding, conversation_id, top_k=5, threshold=SIMILARITY_THRESHOLD):
    """
    Find top-k semantically similar messages (user or AI) 
    from the current conversation only using Python-side similarity.
    """
    if query_embedding is None:
        return []

    # Fetch all messages with embeddings for the specific conversation and user
    query = (
        Message.query.join(Conversation)
        .filter(
            Conversation.user_id == user_id,
            Message.conversation_id == conversation_id,
            Message.embedding.isnot(None)
        )
    )

    messages = query.all()
    results = []

    for msg in messages:
        msg_embedding = msg.get_embedding()
        if msg_embedding is not None:
            similarity = cosine_similarity([query_embedding], [msg_embedding])[0][0]
            if similarity >= threshold:
                results.append((similarity, msg))

    # Sort by highest similarity first
    results.sort(key=lambda x: x[0], reverse=True)
    
    # Return the top_k Message objects
    return [msg for _, msg in results[:top_k]]


# def search_user_messages(user_id, query_embedding, top_k=5, threshold=SIMILARITY_THRESHOLD, conversation_id=None):
#     """
#     Find top-k semantically similar messages (user or AI) from ALL conversations of the user.
#     Returns a list of Message.
#     """
#     if query_embedding is None:
#         return []

#     query = (
#         Message.query.join(Conversation)
#         .filter(Conversation.user_id == user_id, Message.embedding.isnot(None))
#     )
#     if conversation_id:
#         query = query.filter(Message.conversation_id == conversation_id)

#     messages = query.all()

#     results = []
#     for msg in messages:
#         msg_embedding = msg.get_embedding()
#         if msg_embedding is not None:
#             similarity = cosine_similarity([query_embedding], [msg_embedding])[0][0]
#             if similarity >= threshold:  
#                 results.append((similarity, msg))

#     results.sort(key=lambda x: x[0], reverse=True)
#     return [msg for _, msg in results[:top_k]]

def find_similar_answer(user_id, query_embedding, conversation_id, top_k=5, threshold=SIMILARITY_THRESHOLD):
    """
    Find the most semantically similar past user message in the current conversation 
    and return its paired AI answer.
    """
    if query_embedding is None:
        return None

    query = (
        Message.query.join(Conversation)
        .filter(
            Conversation.user_id == user_id,
            Message.is_user == True,
            Message.embedding.isnot(None),
            Message.conversation_id == conversation_id
        )
    )

    user_messages = query.all()
    best_match = None
    best_score = -1

    for msg in user_messages:
        msg_embedding = msg.get_embedding()
        if msg_embedding is not None:
            similarity = cosine_similarity([query_embedding], [msg_embedding])[0][0]
            if similarity > best_score:
                best_score = similarity
                best_match = msg

    # If good match, fetch its AI answer
    if best_match and best_score >= threshold:
        ai_answer = (
            Message.query.filter_by(conversation_id=best_match.conversation_id, is_user=False)
            .filter(Message.timestamp > best_match.timestamp)
            .order_by(Message.timestamp.asc())
            .first()
        )
        if ai_answer:
            return {
                "similarity": best_score,
                "user_message": best_match,
                "ai_answer": ai_answer
            }

    return None

def is_repeated_question(question_embedding, user_id, conversation_id, threshold=SIMILARITY_THRESHOLD):
    """
    Check if a question is similar to previously asked questions.
    If match found, return dict {question, answer, similarity, timestamp, conversation_title}
    """
    if question_embedding is None:
        return None

    if not isinstance(question_embedding, np.ndarray):
        question_embedding = np.array(question_embedding)

    query = (
        Message.query.filter(
            Message.conversation_id == conversation_id,
            Message.is_user == True,
            Message.embedding.isnot(None)
        )
    )
    print("msg from repeated que")

    candidate_msgs = query.all()
    best_match = None
    best_similarity = threshold

    for msg in candidate_msgs:
        msg_embedding = msg.get_embedding()
        if msg_embedding is None:
            continue
        similarity = cosine_similarity([question_embedding], [msg_embedding])[0][0]
        if similarity >= best_similarity:
            # Find the first AI response after this user question
            ai_response = (
                Message.query.filter(
                    Message.conversation_id == msg.conversation_id,
                    Message.is_user == False,
                    Message.timestamp > msg.timestamp
                )
                .order_by(Message.timestamp.asc())
                .first()
            )
            if ai_response:
                conversation = Conversation.query.get(msg.conversation_id)
                best_match = {
                    "question": msg,
                    "answer": ai_response,
                    "similarity": similarity,
                    "timestamp": msg.timestamp,
                    "conversation_title": conversation.title if conversation else "This Conversation",
                }
                best_similarity = similarity

    return best_match


def format_previous_response(question_data, current_question):
    """Format a response indicating this is a repeat question"""
    date_str = question_data["timestamp"].strftime("%B %d, %Y at %H:%M")
    conv_title = question_data["conversation_title"]

    previous_question = question_data["question"].content.strip().lower()
    current_question_lower = current_question.strip().lower()

    if previous_question == current_question_lower:
        return f"You asked this exact question in '{conv_title}' on {date_str}:\n\n{question_data['answer'].content}"
    else:
        return (
            f"You asked a similar question in '{conv_title}' on {date_str}:\n\n"
            f"Previous question: '{question_data['question'].content}'\n\n"
            f"Answer: {question_data['answer'].content}"
        )


# def summarize_history(messages, max_length=1000):
#     """Summarize conversation history if it's too long"""
#     if not messages:
#         return ""
        
#     return "\n".join([f"{'user' if msg.is_user else 'assistant'}: {msg.content}" for msg in messages])
    # total_length = sum(len(msg.content) for msg in messages)
    # if total_length <= max_length:
    # else:
    #     recent_messages = messages[-5:]
    #     summary = "Previous conversation summary: [The conversation was about "

    #     topics = set()
    #     for msg in messages[:-5]:
    #         words = msg.content.lower().split()[:10]
    #         topics.update(words)

    #     summary += ", ".join(list(topics)[:5]) + "].\n"
    #     summary += "\n".join([f"{'User' if msg.is_user else 'AI'}: {msg.content}" for msg in recent_messages])
    #     return summary

def summarize_history(messages, llm, max_turns=30):
    """
    Summarize conversation history dynamically with LLM:
    - Keep the last `max_turns` messages fully.
    - Summarize older ones into a semantic summary using the LLM itself.
    - Produces ChatGPT-style sticky memory (story/game/chat/Q&A aware).
    """
    if not messages:
        return ""

    recent_msgs = messages[-max_turns:]
    old_msgs = messages[:-max_turns]

    def format_msg(msg):
        role = "user" if msg.is_user else "assistant"
        return f"{role}: {msg.content.strip()}"

    recent_text = "\n".join([format_msg(m) for m in recent_msgs])

    if old_msgs:
        old_text = "\n".join([format_msg(m) for m in old_msgs])

        # Use LLM to summarize the old history
        summarization_prompt = f"""
        You are a memory engine. Summarize the following old chat history
        into a compact semantic memory. Capture ongoing tasks, roles,
        and important context (games, Q&A, storytelling, etc.).
        Keep it factual and concise.

        Old conversation:
        {old_text}

        Output format:
        - Mode: (e.g., "story", "Q&A", "game", "chat")
        - Key context: (one short paragraph summary)
        """

        summary_response = llm.invoke(summarization_prompt)
        old_summary = getattr(summary_response, "content", str(summary_response))

        history_summary = (
            f"Earlier conversation summary:\n{old_summary}\n\n"
            f"Recent conversation:\n{recent_text}"
        )
    else:
        history_summary = recent_text

    return history_summary



def calculate_context_similarity(question, context):
    """Calculate how relevant the context is to the question"""
    if not context:
        return 0.0
    
    try:
        question_embedding = get_embedding(question)
        context_embedding = get_embedding(context[:1000])
        
        if question_embedding is None or context_embedding is None:
            return 0.0
        
        similarity = cosine_similarity(
            [question_embedding], 
            [context_embedding]
        )[0][0]
        
        similarity = max(0.0, min(1.0, similarity))
        print(f"Context similarity score: {similarity}")
        return similarity
        
    except Exception as e:
        print(f"Similarity calculation failed: {str(e)}")
        return 0.0


def should_use_web_fallback(question, context_similarity_score, has_relevant_context):
    """Determine if web search fallback should be used.
    Will use web search if the context is not sufficient"""

    question = (question or "").lower()

    SCORE = Config.SCORE_OF_SIMILARITY
    
    if context_similarity_score < SCORE or not has_relevant_context:
        return True
    
    return 
    



def enhanced_prompt_template(history, context, user_input, web_context=None, modelSize="small"):
    """
    Generates a structured list of messages with roles (system, user) and content.
    """

    # Choose the appropriate prompt template based on model size
    if modelSize == "small":
        return prompt_for_small_model(history, context, user_input, web_context, 4)

    # bellow is for large models
    # The system message defines the AI's persona, rules, and core directives.
    system_prompt = """
You are 'Assistant', a friendly, empathetic, and highly capable AI. Your goal is to provide natural, varied, and helpful responses.

**SOURCE OF TRUTH HIERARCHY:**
You MUST follow this order to understand the context of the conversation.
1.  **DOCUMENT CONTEXT:** If the user's message relates to specific knowledge, prioritize this source.
2.  **CONVERSATION HISTORY:** Use this to understand the flow of the chat and recall previous points.
3.  **WEB SEARCH RESULTS:** Use this for recent or external information if relevant.
4.  **GENERAL KNOWLEDGE:** If no provided context is relevant, respond based on your general knowledge.

**STYLE AND TONE RULES:**
- **Be Conversational:** Keep your tone warm and natural. Use emojis sparingly to add personality. 🧑‍💻
- **Never Repeat:** Do NOT give the exact same response twice. Always rephrase, use an analogy, or add new details.
- **Handle Greetings:** When a user says 'Hi' or 'Hello', keep your response to 2-3 sentences. Greet them back warmly and ask how you can assist. Vary your greeting each time.
- **Use Context Naturally:** Seamlessly blend information from the context into your response. Do not just copy-paste. Summarize and explain in your own words.
- **We directly sending your response to user, so send only what user need to see.
"""

    # The user message contains all the dynamic data for the current turn.
    # All context and the latest user input are combined here under the 'user' role.
    user_content_parts = [
        "<CONVERSATION_HISTORY>",
        history,
        "</CONVERSATION_HISTORY>",
        "\n",
        "<DOCUMENT_CONTEXT>",
        context,
        "</DOCUMENT_CONTEXT>",
    ]

    if web_context:
        user_content_parts.extend([
            "\n",
            "<WEB_SEARCH_RESULTS>",
            web_context,
            "</WEB_SEARCH_RESULTS>",
        ])

    user_content_parts.extend([
        "\n\n",
        "Considering all the information above, provide a natural and relevant response to the user's latest message.",
        "\n<USER_INPUT>",
        user_input,
        "</USER_INPUT>"
    ])
    
    user_content = "".join(user_content_parts)

    # The final output is a list of dictionaries, the standard format.
    merged_prompt = f"{system_prompt}\n\n{user_content}"

    return [{'role': 'user', 'content': merged_prompt}]

 


def prompt_for_small_model(history, context, user_input, web_context=None, max_history_turns=2):
    """
    Generates a simplified, multi-turn chat prompt suitable for small models (2B-8B).
    It uses a direct system prompt and a clean chat history format.
    """
    
    # 1. A drastically simplified system prompt.
    simplified_system_prompt = "You are a friendly and helpful assistant. Use the provided context to answer the user's message. Be concise and do not repeat yourself."

    messages = []
    
    # 2. Process the history string into a clean, turn-by-turn list.
    
    turns = history.strip().split('\n') if history else []
    current_user_line = f"user: {user_input.strip().lower()}"
    if turns and turns[-1].strip().lower() == current_user_line:
        turns.pop()


    print("context:", context)
    print("++")
    print("history:", history)
    print("++")

    chat_history = []
    for turn in turns:
        if turn.startswith("user:"):
            chat_history.append({'role': 'user', 'content': turn.replace("user:", "").strip()})
        elif turn.startswith("assistant:"):
            chat_history.append({'role': 'assistant', 'content': turn.replace("assistant:", "").strip()})
    
    # Only keep the last few turns
    messages.extend(chat_history[-max_history_turns*2:])
    # messages.extend(chat_history)
    # # remove current user message
    # if messages:
    #     messages.pop()
    # 3. Construct the final user message, prioritizing context.
    final_user_parts = [simplified_system_prompt + "\n\n"]

    if context.strip():
        final_user_parts.append(f"Use this context to answer:\n---\n{context}\n---\n")
    elif web_context and web_context.strip():
        final_user_parts.append(f"Use this web information to answer:\n---\n{web_context}\n---\n")
    
    final_user_parts.append(user_input)
    final_user_content = "".join(final_user_parts)
    
    messages.append({'role': 'user', 'content': final_user_content})
    

    return messages

prompt_template = ChatPromptTemplate.from_template("""
You are an expert assistant trained to answer various questions based on provided context and conversation history.

Conversation History:
{history}

Relevant Context from Documents:
{context}

Current Question: {question}

Please provide a helpful answer based on the conversation history and document context:
""")

def enhance_with_mcp(question, context, conversation_messages, question_embedding, initial_response="", web_context=""):  
    """
    Use MCP to enhance context and improve response quality.
    """
    try:
        # Convert messages into the format MCP expects
        mcp_messages = []
        for msg in conversation_messages:
            content = getattr(msg, 'content', '') if hasattr(msg, 'content') else msg.get('content', '')
            is_user = getattr(msg, 'is_user', False) if hasattr(msg, 'is_user') else msg.get('is_user', False)
            emb = None
            if hasattr(msg, 'get_embedding'):
                emb_val = msg.get_embedding()
                if emb_val is not None:
                    emb = emb_val.tolist() if hasattr(emb_val, 'tolist') else list(emb_val)
            elif isinstance(msg, dict):
                emb = msg.get('embedding')
            mcp_messages.append({
                'id': getattr(msg, 'id', None) if hasattr(msg, 'id') else msg.get('id'),
                'content': summarize_text(content, max_sentences=3, max_chars=400),  # compact each msg
                'is_user': is_user,
                'embedding': emb
            })

        # Summarize the conversation context itself if it's too long
        condensed_context = summarize_text(context, max_sentences=6, max_chars=1200)

        # Call MCP for analysis with web_context
        mcp_result = mcp_client.analyze_and_enhance(
            user_question=question,
            context=condensed_context,
            conversation_messages=mcp_messages,
            question_embedding=question_embedding.tolist() if hasattr(question_embedding, 'tolist') else question_embedding,
            initial_llm_response=initial_response,
            web_context=web_context  
        )

        print(f"MCP Enhancement: {mcp_result.get('recommendation')}, {mcp_result['search_results']['total_matches']} matches")
        return mcp_result

    except Exception as e:
        print(f"MCP enhancement error: {e}")
        return {
            "needs_enhancement": False,
            "enhanced_context": context,
            "keywords": [],
            "search_results": {"relevant_messages": [], "search_method": "error", "total_matches": 0},
            "confidence_score": 0.5,
            "recommendation": "use_original"
        }

def _clean_final_response(response):
    """Clean the final response to remove internal reasoning artifacts"""
    if not response:
        return response
    
    # Remove common reasoning prefixes
    reasoning_indicators = [
        "Step 1:", "Step 2:", "Step 3:", "Step 4:", "Step 5:",
        "Reasoning:", "Analysis:", "Final Answer:", "Answer:",
        "[general knowledge]", "(general knowledge)"
    ]
    
    cleaned = response
    for indicator in reasoning_indicators:
        if indicator in cleaned:
            parts = cleaned.split(indicator)
            if len(parts) > 1:
                cleaned = parts[-1].strip()
    
    lines = cleaned.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        if (line.startswith('- ') or 
            re.match(r'^\d+\.\s', line) or
            line.startswith('• ') or
            any(indicator in line for indicator in ['Relevant facts', 'Step-by-step', 'What\'s missing'])):
            continue
        clean_lines.append(line)
    
    result = ' '.join(clean_lines).strip()
    
    if not result:
        return response
    
    # # Ensure the response doesn't start with reasoning markers
    # if any(result.startswith(marker) for marker in ['1.', '2.', '3.', '4.', '5.']):
    #     # Find the first non-numbered part
    #     match = re.search(r'[^0-9\.](.*)', result)
    #     if match:
    #         result = match.group(1).strip()
    
    return result
    
def self_repair_llm_response(llm, question, original_context, web_context, conversation_messages, question_embedding, initial_response):
    """
    Self-repair mechanism using MCP enhancement.
    """
    print("Activating MCP self-repair mechanism...")

    mcp_result = enhance_with_mcp(
        question=question,
        context=original_context,
        conversation_messages=conversation_messages,
        question_embedding=question_embedding,
        initial_response=initial_response,
        web_context=web_context
    )

    # If no enhancement recommended, return original answer
    if not mcp_result.get("needs_enhancement", False) or not mcp_result.get("enhanced_context"):
        print("MCP determined no enhancement needed")
        return initial_response, False

    # Condense the enhanced_context further to keep prompt size reasonable
    enhanced_context = limit_text(mcp_result["enhanced_context"], max_chars=1500)

    # FIXED: Better repair prompt that produces clean final answers
    repair_prompt = f"""
You are an expert assistant. Re-evaluate the previous answer using the enhanced context below.

ENHANCED CONTEXT (condensed):
{enhanced_context}

USER QUESTION:
{question}

INITIAL RESPONSE (for reference):
{initial_response}

IMPORTANT INSTRUCTIONS:
- Analyze the enhanced context carefully
- If the context contains the answer, use it directly
- If the context is insufficient, supplement with your knowledge
- Provide ONLY the final answer - no reasoning steps, no bullet points, no labels
- Make the response natural and conversational
- Do NOT mention "[general knowledge]" or show internal thinking
- Do NOT repeat the user's question in your response
- For greetings like "Hi", keep responses brief and friendly (1-2 sentences)

FINAL ANSWER (natural conversation only):
"""
    try:
        response = llm.invoke(repair_prompt)
        repaired_response = getattr(response, "content", str(response))
        
        # Clean up the response - remove any internal reasoning that might have leaked through
        cleaned_response = _clean_final_response(repaired_response)
        
        print("MCP self-repair completed successfully")
        return cleaned_response, True
    except Exception as e:
        print(f"Self-repair failed: {e}")
        return initial_response, False


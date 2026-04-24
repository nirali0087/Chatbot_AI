import os
import json
import numpy as np
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import requests
import re

class MCPServer:
    """
    Model Context Protocol Server - Intelligent middleware for LLM reasoning
    """
    
    def __init__(self, host='localhost', port=5001):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Confidence thresholds
        self.confidence_threshold = 0.6
        self.similarity_threshold = 0.7
        
    def setup_routes(self):
        """Setup MCP server endpoints"""
        @self.app.route('/mcp/analyze', methods=['POST'])
        def analyze_and_enhance():
            return self.analyze_and_enhance_request()
            
        @self.app.route('/mcp/health', methods=['GET'])
        def health_check():
            return jsonify({"status": "healthy", "service": "MCP Server"})
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from user question"""
        if not text:
            return []
            
        # Remove common stop words and extract meaningful keywords
        stop_words = {'what', 'how', 'why', 'when', 'where', 'which', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for word in keywords:
            if word not in seen:
                seen.add(word)
                unique_keywords.append(word)
                
        return unique_keywords[:6]  # Return top 10 keywords
    
    def calculate_confidence(self, llm_response: str, question: str, context: str) -> float:
        """
        Calculate confidence score for LLM response
        Returns: confidence score between 0 and 1
        """
        try:
            confidence_indicators = 0
            total_indicators = 0
            
            # Check for uncertainty phrases
            uncertainty_phrases = [
                "i'm not sure", "i don't know", "i'm not certain", 
                "i think", "maybe", "perhaps", "possibly",
                "i'm not entirely sure", "i could be wrong"
            ]
            
            response_lower = llm_response.lower()
            for phrase in uncertainty_phrases:
                if phrase in response_lower:
                    confidence_indicators -= 1
                total_indicators += 1
            
            # Check for definitive answers
            definitive_phrases = [
                "definitely", "certainly", "absolutely", "without a doubt",
                "the answer is", "clearly", "obviously"
            ]
            
            for phrase in definitive_phrases:
                if phrase in response_lower:
                    confidence_indicators += 1
                total_indicators += 1
            
            # Check response length (too short might indicate uncertainty)
            if len(llm_response.strip().split()) < 10:
                confidence_indicators -= 1
            total_indicators += 1
            
            # Normalize confidence score
            confidence = max(0.0, min(1.0, 0.5 + (confidence_indicators / max(1, total_indicators)) * 0.5))
            
            print(f"MCP Confidence Score: {confidence}")
            return confidence
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.5  # Default medium confidence
    
    def search_conversation_messages(self, conversation_messages: List[Dict], keywords: List[str], question_embedding: List[float]) -> Dict[str, Any]:
        """
        Search through conversation messages using keywords and semantic similarity
        """
        relevant_messages = []
        
        if not conversation_messages:
            return {"relevant_messages": [], "search_method": "none"}
        
        # Method 1: Keyword-based search
        keyword_matches = []
        for msg in conversation_messages:
            content = msg.get('content', '').lower()
            msg_keywords = self.extract_keywords(content)
            
            # Calculate keyword overlap
            overlap = len(set(keywords) & set(msg_keywords))
            if overlap > 0:
                keyword_matches.append({
                    'message': msg,
                    'overlap_score': overlap / len(keywords) if keywords else 0,
                    'method': 'keyword'
                })
        
        # Method 2: Semantic search (if embedding available)
        semantic_matches = []
        if question_embedding and len(question_embedding) > 0:
            for msg in conversation_messages:
                if msg.get('embedding'):
                    try:
                        msg_embedding = msg['embedding']
                        if isinstance(msg_embedding, list) and len(msg_embedding) > 0:
                            similarity = cosine_similarity([question_embedding], [msg_embedding])[0][0]
                            if similarity > self.similarity_threshold:
                                semantic_matches.append({
                                    'message': msg,
                                    'similarity_score': similarity,
                                    'method': 'semantic'
                                })
                    except Exception as e:
                        print(f"Error in semantic similarity: {e}")
                        continue
        
        # Combine and rank results
        all_matches = keyword_matches + semantic_matches
        
        # Sort by score (either overlap_score or similarity_score)
        def get_score(match):
            return match.get('overlap_score', 0) or match.get('similarity_score', 0)
        
        all_matches.sort(key=get_score, reverse=True)
        
        # Take top 5 most relevant messages
        top_matches = all_matches[:5]
        relevant_messages = [match['message'] for match in top_matches]
        
        search_method = "combined"
        if keyword_matches and not semantic_matches:
            search_method = "keyword"
        elif semantic_matches and not keyword_matches:
            search_method = "semantic"
        
        return {
            "relevant_messages": relevant_messages,
            "search_method": search_method,
            "total_matches": len(all_matches)
        }
    
    def build_enhanced_context(self, original_context: str, search_results: Dict, keywords: List[str], web_context: str = "") -> str:  # MODIFIED
        """Build enhanced context from search results and web context"""
        enhanced_parts = []
        
        # Add original context if exists
        if original_context and original_context.strip():
            enhanced_parts.append("ORIGINAL CONTEXT:")
            enhanced_parts.append(original_context)
        
        # Add web context if available
        if web_context and web_context.strip():
            enhanced_parts.append("\nWEB SEARCH RESULTS:")
            enhanced_parts.append(web_context)
        
        # Add conversation history insights
        relevant_messages = search_results.get('relevant_messages', [])
        if relevant_messages:
            enhanced_parts.append("\nRELEVANT CONVERSATION HISTORY:")
            for i, msg in enumerate(relevant_messages, 1):
                role = "USER" if msg.get('is_user') else "ASSISTANT"
                content = msg.get('content', '')
                enhanced_parts.append(f"{i}. {role}: {content}")
        
        # Add keyword insights
        if keywords:
            enhanced_parts.append(f"\nKEY TOPICS: {', '.join(keywords)}")
        
        # Add search methodology
        search_method = search_results.get('search_method', 'none')
        enhanced_parts.append(f"\nCONTEXT ENHANCEMENT: Used {search_method} search, found {len(relevant_messages)} relevant messages")
        
        return "\n".join(enhanced_parts)
    
    def analyze_and_enhance_request(self):
        """
        Main MCP endpoint for analyzing and enhancing LLM requests
        """
        try:
            data = request.get_json()
            
            # Extract request components
            user_question = data.get('user_question', '')
            original_context = data.get('context', '')
            conversation_messages = data.get('conversation_messages', [])
            question_embedding = data.get('question_embedding', [])
            initial_llm_response = data.get('initial_llm_response', '')
            web_context = data.get('web_context', '')  # ADD THIS LINE
            
            print(f"MCP Processing: '{user_question[:50]}...'")
            
            # Step 1: Extract keywords
            keywords = self.extract_keywords(user_question)
            print(f"Extracted keywords: {keywords}")
            
            # Step 2: If we have initial LLM response, check confidence
            if initial_llm_response:
                confidence = self.calculate_confidence(initial_llm_response, user_question, original_context)
                needs_enhancement = confidence < self.confidence_threshold
            else:
                confidence = 0.5  # Default medium confidence
                needs_enhancement = True
            
            # Step 3: Search conversation messages
            search_results = self.search_conversation_messages(conversation_messages, keywords, question_embedding)
    
            # Step 3.5: Check for direct match with previous Q&A
            matched_answer = None
            high_confidence_threshold = max(self.similarity_threshold + 0.15, 0.85)
            
            for match in search_results.get('relevant_messages', []):
                if not match.get('is_user'):  # Only consider assistant answers
                    msg_embedding = match.get('embedding')
                    if msg_embedding and question_embedding:
                        try:
                            similarity = cosine_similarity([question_embedding], [msg_embedding])[0][0]
                            if similarity >= high_confidence_threshold:
                                matched_answer = match.get('content')
                                print(f"Found high-confidence matched answer: {matched_answer}")
                                break
                        except Exception as e:
                            print(f"Error comparing embeddings: {e}")
            
            # Step 4: Build enhanced context (MODIFY THIS FUNCTION)
            enhanced_context = self.build_enhanced_context(original_context, search_results, keywords, web_context)  # MODIFIED
            
            # Step 5: Determine if we should proceed with enhancement
            should_enhance = needs_enhancement or search_results['total_matches'] > 0
            
            response_data = {
                "needs_enhancement": should_enhance,
                "enhanced_context": enhanced_context,
                "keywords": keywords,
                "search_results": search_results,
                "confidence_score": confidence,
                "recommendation": "enhance_and_retry" if should_enhance else "use_original"
            }
    
            if matched_answer:
                response_data["recommendation"] = "reuse_previous_answer"
                response_data["matched_answer"] = matched_answer
            
            print(f"MCP Decision: {'ENHANCE' if should_enhance else 'USE ORIGINAL'}")
            return jsonify(response_data)
            
        except Exception as e:
            print(f"MCP Error: {e}")
            return jsonify({
                "needs_enhancement": False,
                "enhanced_context": "",
                "keywords": [],
                "search_results": {"relevant_messages": [], "search_method": "error", "total_matches": 0},
                "confidence_score": 0.5,
                "recommendation": "use_original",
                "error": str(e)
            })
                
    
    def run(self):
        """Start the MCP server"""
        print(f"Starting MCP Server on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=False)

mcp_server = MCPServer()

if __name__ == '__main__':
    mcp_server.run()

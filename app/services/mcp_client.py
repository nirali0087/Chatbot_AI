import requests
import json
from typing import Dict, List, Optional, Any


class MCPClient:
    """
    MCP Client for communicating with the MCP Server
    """
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.health_check()
    
    def health_check(self) -> bool:
        """Check if MCP server is healthy"""
        try:
            response = requests.get(f"{self.base_url}/mcp/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            print("MCP Server is not available. Running without enhancement.")
            return False
    
    def analyze_and_enhance(
        self, 
        user_question: str,
        context: str,
        conversation_messages: List[Dict],
        question_embedding: List[float],
        initial_llm_response: str = "",
        web_context: str = ""
    ) -> Dict[str, Any]:
        """
        Send request to MCP server for analysis and enhancement
        """
        if not self.health_check():
            # Return default response if MCP server is down
            return {
                "needs_enhancement": False,
                "enhanced_context": context,
                "keywords": [],
                "search_results": {"relevant_messages": [], "search_method": "offline", "total_matches": 0},
                "confidence_score": 0.7,
                "recommendation": "use_original"
            }
        
        try:
            payload = {
                "user_question": user_question,
                "context": context,
                "conversation_messages": conversation_messages,
                "question_embedding": question_embedding,
                "initial_llm_response": initial_llm_response,
                "web_context": web_context
            }
            
            response = requests.post(
                f"{self.base_url}/mcp/analyze",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"MCP Server error: {response.status_code}")
                return {
                    "needs_enhancement": False,
                    "enhanced_context": context,
                    "keywords": [],
                    "search_results": {"relevant_messages": [], "search_method": "error", "total_matches": 0},
                    "confidence_score": 0.7,
                    "recommendation": "use_original"
                }
                
        except requests.exceptions.RequestException as e:
            print(f"MCP Client error: {e}")
            return {
                "needs_enhancement": False,
                "enhanced_context": context,
                "keywords": [],
                "search_results": {"relevant_messages": [], "search_method": "offline", "total_matches": 0},
                "confidence_score": 0.7,
                "recommendation": "use_original"
            }

mcp_client = MCPClient() 
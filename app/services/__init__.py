from .vector_utils import (
    initialize_vector_store, initialize_llm, get_embedding,
    search_user_messages, find_similar_answer, is_repeated_question,
    format_previous_response, summarize_history, calculate_context_similarity, should_use_web_fallback,
    enhanced_prompt_template
)

from app.services.web_search import web_search_fallback, format_web_results

__all__ = [
    'initialize_vector_store', 'initialize_llm', 'get_embedding',
    'search_user_messages', 'find_similar_answer', 'is_repeated_question',
    'format_previous_response', 'summarize_history', 'web_search_fallback',
    'format_web_results', 'calculate_context_similarity', 'should_use_web_fallback',
    'enhanced_prompt_template'
]
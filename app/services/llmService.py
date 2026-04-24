from app.config import Config
import litellm


def chat_completion(messages):
    """Sends a prompt to the specified LLM via LiteLLM and gets a response."""
    try:

        # "openai" prefix for openai style api
        model = "openai/" + Config.LM_STUDIO_LLM_MODEL
        # The core LiteLLM function
        response = litellm.completion(
            api_base=Config.LM_STUDIO_BASE_URL,
            api_key="not-needed",
            model=model,
            messages=messages,
            temperature=0.1,
            # max_tokens=500,
            # stream=False
        )

        # Extract the content from the response
        return response.choices[0].message.content

    except Exception as e:
        # LiteLLM provides detailed exception logging
        print(f"LiteLLM Error: {e}")
        return "Sorry, I'm having trouble connecting to my brain right now."

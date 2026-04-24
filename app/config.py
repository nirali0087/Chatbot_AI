import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'fallback-secret-key')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
    DB_NAME = os.getenv('DB_NAME', 'chatbot_db')
    DB_USER = os.getenv('DB_USER', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DB_PORT = os.getenv('DB_PORT', '3306')
    
    # For MySQL, use the appropriate dialect + driver
    # Example with PyMySQL:
    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False


    # # print(9876543))
    # SQLALCHEMY_TRACK_MODIFICATIONS = False
    print(234567)

    
    # LM Studio Configuration
    LM_STUDIO_BASE_URL = os.getenv('LM_STUDIO_BASE_URL', 'http://localhost:1234/v1')
    # LM_STUDIO_EMBEDDING_MODEL = os.getenv('LM_STUDIO_EMBEDDING_MODEL', 'gaianet/Nomic-embed-text-v1.5-Embedding-GGUF')
    # LM_STUDIO_LLM_MODEL = os.getenv('LM_STUDIO_LLM_MODEL', 'google/gemma-3-1b')
    LM_STUDIO_LLM_MODEL = os.getenv('LM_STUDIO_LLM_MODEL', 'mistral-7b-instruct-v0.2')
    
    OLLAMA_EMBEDDING_MODEL = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
    OLLAMA_EMBEDDING_URL = os.getenv('OLLAMA_EMBEDDING_URL', 'http://localhost:11434')

    # GPU Configuration
    USE_GPU = os.getenv('USE_GPU', 'True').lower() == 'true'
    GPU_DEVICE_ID = int(os.getenv('GPU_DEVICE_ID', '0'))
    MAX_GPU_MEMORY = float(os.getenv('MAX_GPU_MEMORY', '0.9'))
  
    FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'pdf_faiss_index')
    CONTEXT_ENABLE = os.getenv('CONTEXT_ENABLE', 'True').lower() == 'true'


    SCORE_OF_SIMILARITY = float(os.getenv("SCORE_OF_SIMILARITY", "0.4"))
    
    WEB_SEARCH_ENABLED = os.getenv('WEB_SEARCH_ENABLED', 'True').lower() == 'true'
    WEB_SEARCH_RESULTS = int(os.getenv('WEB_SEARCH_RESULTS', '3'))
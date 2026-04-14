import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (parent of src/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    ENDEE_API_KEY = os.getenv("ENDEE_API_KEY", "local")
    ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")

    # Chunking
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Embedding
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384

    # Search
    TOP_K_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.3

    # LLM
    GROQ_MODEL = "llama-3.3-70b-versatile"
    MAX_TOKENS = 1024
    TEMPERATURE = 0.1

    # Collection
    COLLECTION_NAME = "endee_rag"

    # History
    MAX_HISTORY_TURNS = 5
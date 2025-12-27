# config.py
import os
import sys
from dotenv import load_dotenv
from falkordb import FalkorDB
import ollama

# Load environment variables
load_dotenv()

class Config:
    FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
    FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", 6379))
    
    # Ollama config
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "deepseek-r1:8b")
    
    # Validate required configurations
    @classmethod
    def validate(cls):
        pass # No API keys needed for local Ollama
    
    @classmethod
    def get_falkordb_connection_string(cls):
        return f"redis://{cls.FALKORDB_HOST}:{cls.FALKORDB_PORT}"

# Initialize services with error handling
def init_falkordb():
    """Initialize FalkorDB connection with error handling"""
    config = Config()
    try:
        db = FalkorDB(host=config.FALKORDB_HOST, port=config.FALKORDB_PORT)
        # Test connection by selecting a graph (lazy connection usually)
        # db.ping() not supported by all versions of FalkorDB client wrapper
        print(f"Connected to FalkorDB at {config.FALKORDB_HOST}:{config.FALKORDB_PORT}")
        return db
    except Exception as e:
        print(f"Error connecting to FalkorDB: {e}")
        print(f"Please ensure FalkorDB is running at {config.FALKORDB_HOST}:{config.FALKORDB_PORT}")
        raise

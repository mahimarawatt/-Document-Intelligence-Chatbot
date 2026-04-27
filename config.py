# All configurable settings live here — easy to change without touching main code

GROQ_MODEL = "llama-3.3-70b-versatile"   # Free, fast, powerful
EMBEDDING_MODEL = "all-MiniLM-L6-v2"     # Local HuggingFace model, no API needed
CHUNK_SIZE = 1000        # How many characters per text chunk
CHUNK_OVERLAP = 200      # Overlap between chunks to preserve context
TOP_K_RESULTS = 4        # How many relevant chunks to retrieve
CHROMA_DIR = "./chroma_db"  # Where ChromaDB saves vectors on disk
MAX_HISTORY = 6          # Number of past messages to keep for context
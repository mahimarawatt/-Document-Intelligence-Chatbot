# 📄 Document Intelligence Chatbot

A RAG (Retrieval-Augmented Generation) powered chatbot that lets you upload documents and ask questions about them using AI. Instead of fine-tuning a model on your data, it retrieves the most relevant chunks from your documents at query time and feeds them to the LLM — giving accurate, document-grounded answers.

---

## ✨ Features

- 📁 Upload PDF, TXT, or DOCX files
- 🧠 Semantic search using local HuggingFace embeddings (no API cost)
- 💬 Conversational Q&A with memory (remembers last 6 turns)
- 🔍 Source transparency — see exactly which document chunks were used
- 💾 Persistent vector store — survives app restarts
- 📚 Multi-document support — index and query multiple files together

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq API (Llama 3.3 70B) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (runs locally) |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |

---

## 📁 Project Structure

```
document_intelligence_chatbot/
├── app.py                  # Streamlit UI — run this to start the app
├── rag_pipeline.py         # Core RAG logic (chunk, embed, store, retrieve, generate)
├── document_processor.py   # File parsing for PDF, TXT, and DOCX
├── config.py               # All configuration constants in one place
├── requirements.txt        # Python dependencies
└── chroma_db/              # Vector store — auto-created on first document upload
```

---

## ✅ Prerequisites

- Python 3.10 or higher
- A free [Groq API key](https://console.groq.com) 

---

## 🚀 Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/document-intelligence-chatbot.git
cd document-intelligence-chatbot
```

**2. Create a virtual environment**

```bash
# Using pip
python -m venv .venv

# Activate — Windows PowerShell
.venv\Scripts\activate

# Activate — macOS/Linux
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
pip install langchain-classic langchain-core langchain-text-splitters langchain-huggingface langchain-chroma
```

**4. Create your `.env` file**

Create a file named `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free key at [https://console.groq.com](https://console.groq.com)

---

## ⚙️ Configuration

All settings are in `config.py`:

```python
GROQ_MODEL = "llama-3.3-70b-versatile"  # LLM model to use
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # Local embedding model
CHUNK_SIZE = 1000       # Characters per text chunk
CHUNK_OVERLAP = 200     # Overlap between chunks
TOP_K_RESULTS = 4       # Number of chunks retrieved per query
MAX_HISTORY = 6         # Conversation turns the chatbot remembers
CHROMA_DIR = "./chroma_db"
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 💬 How to Use

1. **Upload documents** — use the sidebar to upload PDF, TXT, or DOCX files
2. **Process documents** — click **🚀 Process Documents** to index them
3. **Ask questions** — type in the chat input at the bottom
4. **View sources** — expand **📎 source chunks** under any answer
5. **Clear data** — click **🗑️ Clear All Data** to start fresh

---

## 🏗 How It Works

```
Your Document
     │
     ▼
[Document Processor]  ← Extracts raw text from PDF / TXT / DOCX
     │
     ▼
[Text Splitter]       ← Breaks text into overlapping chunks (1000 chars, 200 overlap)
     │
     ▼
[Embedding Model]     ← Converts each chunk into a vector (runs locally, free)
     │
     ▼
[ChromaDB]            ← Stores vectors on disk for fast similarity search
     │
     ▼
  User Query
     │
     ▼
[Retriever]           ← Finds top-4 most relevant chunks via semantic search
     │
     ▼
[Groq LLM]            ← Generates answer grounded in retrieved chunks
     │
     ▼
  Chat Response + Source Chunks
```

## 📄 License

MIT License — free to use, modify, and distribute.

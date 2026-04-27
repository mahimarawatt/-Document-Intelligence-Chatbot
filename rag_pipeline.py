# This is where the RAG magic happens:
# 1. Split documents into chunks
# 2. Embed chunks and store in ChromaDB
# 3. On a query: retrieve relevant chunks, pass to LLM with context

import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document

import config

load_dotenv()  # Loads GROQ_API_KEY from .env file


class RAGPipeline:
    """
    Encapsulates the entire RAG workflow.
    
    RAG = Retrieval-Augmented Generation
    Instead of asking the LLM to "remember" your document,
    we RETRIEVE the most relevant chunks at query time
    and AUGMENT the LLM's prompt with that context.
    This way the LLM gives accurate, document-grounded answers.
    """
    
    def __init__(self):
        # Step A: Initialize the embedding model
        # This runs LOCALLY — no API call, no cost.
        # It converts text chunks into numerical vectors (embeddings).
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},   # Use "cuda" if you have a GPU
            encode_kwargs={"normalize_embeddings": True}  # Better similarity scores
        )
        
        # Step B: Initialize the LLM via Groq (free API)
        # Groq runs Llama 3.3 on their own hardware — blazing fast inference.
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=config.GROQ_MODEL,
            temperature=0.2,      # Lower = more factual, less creative
            max_tokens=1024
        )
        
        # Step C: Text splitter — breaks documents into overlapping chunks
        # Why overlap? So that context at chunk boundaries isn't lost.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            # Try to split on paragraphs, then sentences, then words
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.vector_store = None
        self.qa_chain = None
    
    
    def ingest_document(self, text: str, source_name: str) -> int:
        """
        Takes raw document text, splits it, embeds it, stores in ChromaDB.
        Returns the number of chunks created.
        
        This is the INDEXING phase of RAG.
        """
        # Wrap in LangChain Document objects (they carry metadata too)
        doc = Document(page_content=text, metadata={"source": source_name})
        
        # Split into chunks
        chunks = self.text_splitter.split_documents([doc])
        
        if not chunks:
            raise ValueError("Document produced no text chunks. It may be empty or image-only.")
        
        # Create or update ChromaDB vector store
        # persist_directory makes it save to disk — survives app restarts
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=config.CHROMA_DIR
            )
        else:
            # If store already exists, just add new chunks to it
            self.vector_store.add_documents(chunks)
        
        # Build/rebuild the QA chain with the updated store
        self._build_qa_chain()
        
        return len(chunks)
    
    
    def _build_qa_chain(self):
        """
        Assembles the ConversationalRetrievalChain.
        
        This chain does the following on each query:
        1. Rephrase the query using conversation history (so pronouns like "it", "they" resolve correctly)
        2. Run semantic search on ChromaDB to find top-K relevant chunks
        3. Pass the query + retrieved chunks + history to the LLM
        4. Return the LLM's answer
        """
        retriever = self.vector_store.as_retriever(
            search_type="similarity",         # Could also be "mmr" for diversity
            search_kwargs={"k": config.TOP_K_RESULTS}
        )
        
        # Memory keeps last N turns of conversation
        memory = ConversationBufferWindowMemory(
            k=config.MAX_HISTORY,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,   # We'll show sources in the UI
            verbose=False
        )
    
    
    def query(self, question: str) -> dict:
        """
        The GENERATION phase of RAG.
        Takes a user question, retrieves context, generates an answer.
        
        Returns:
            {
                "answer": str,
                "sources": list of source chunk content,
                "num_sources": int
            }
        """
        if self.qa_chain is None:
            raise RuntimeError("No documents ingested yet. Please upload a document first.")
        
        result = self.qa_chain({"question": question})
        
        # Extract the source chunks that were retrieved
        sources = [doc.page_content[:200] + "..." 
                   for doc in result.get("source_documents", [])]
        
        return {
            "answer": result["answer"],
            "sources": sources,
            "num_sources": len(sources)
        }
    
    
    def load_existing_store(self) -> bool:
        """
        Loads a previously saved ChromaDB from disk.
        Returns True if found, False if this is a fresh start.
        """
        try:
            self.vector_store = Chroma(
                persist_directory=config.CHROMA_DIR,
                embedding_function=self.embeddings
            )
            # Check if it's actually populated
            if self.vector_store._collection.count() > 0:
                self._build_qa_chain()
                return True
        except Exception:
            pass
        return False
    
    
    def clear_store(self):
        import shutil, gc, time

        self.qa_chain = None

        if self.vector_store is not None:
            try:
                self.vector_store._client.close()
            except Exception:
                pass
            self.vector_store = None

        gc.collect()
        time.sleep(0.5)

        if os.path.exists(config.CHROMA_DIR):
            shutil.rmtree(config.CHROMA_DIR)
# The main UI — run this with: streamlit run app.py

import streamlit as st
from rag_pipeline import RAGPipeline
from document_processor import load_document

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocBot — RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("📄 Document Intelligence Chatbot")
st.markdown("Upload any PDF, TXT, or DOCX — then ask questions about it using AI.")

# ── Initialize session state ─────────────────────────────────────────────────
# Streamlit reruns the script on every interaction.
# session_state persists variables across reruns.

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()
    # Try to load existing vector store from disk
    if st.session_state.rag.load_existing_store():
        st.session_state.store_loaded = True
    else:
        st.session_state.store_loaded = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # List of (role, message) tuples

if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

# ── Sidebar — Document Upload ─────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="You can upload multiple documents — all will be indexed together."
    )
    
    if uploaded_files:
        if st.button("🚀 Process Documents", use_container_width=True):
            for file in uploaded_files:
                if file.name in st.session_state.doc_names:
                    st.warning(f"Already processed: {file.name}")
                    continue
                
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        # Step 1: Extract text from the file
                        text = load_document(file)
                        
                        if len(text.strip()) < 100:
                            st.error(f"{file.name}: Too little text extracted. May be a scanned image PDF.")
                            continue
                        
                        # Step 2: Ingest into RAG pipeline (embed + store)
                        num_chunks = st.session_state.rag.ingest_document(text, file.name)
                        
                        st.session_state.doc_names.append(file.name)
                        st.success(f"✅ {file.name}: {num_chunks} chunks indexed")
                        st.session_state.store_loaded = True
                    
                    except Exception as e:
                        st.error(f"Error processing {file.name}: {str(e)}")
    
    # Show indexed documents
    if st.session_state.doc_names:
        st.divider()
        st.subheader("📋 Indexed Documents")
        for name in st.session_state.doc_names:
            st.markdown(f"• {name}")
    
    # Clear everything
    st.divider()
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.rag.clear_store()
        st.session_state.chat_history = []
        st.session_state.doc_names = []
        st.session_state.store_loaded = False
        st.success("Cleared!")
        st.rerun()


# ── Main Chat Interface ───────────────────────────────────────────────────────

# Display all past messages
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# Chat input box (appears at bottom of screen)
if question := st.chat_input("Ask a question about your documents..."):
    
    if not st.session_state.store_loaded:
        st.warning("⬅️ Please upload and process a document first using the sidebar.")
        st.stop()
    
    # Show user's message immediately
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_history.append(("user", question))
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag.query(question)
                answer = result["answer"]
                
                st.markdown(answer)
                
                # Show source chunks in an expander (great for transparency)
                if result["sources"]:
                    with st.expander(f"📎 {result['num_sources']} source chunks used"):
                        for i, src in enumerate(result["sources"], 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(src)
                            st.divider()
                
                st.session_state.chat_history.append(("assistant", answer))
            
            except Exception as e:
                err_msg = f"Error: {str(e)}"
                st.error(err_msg)
                st.session_state.chat_history.append(("assistant", err_msg))
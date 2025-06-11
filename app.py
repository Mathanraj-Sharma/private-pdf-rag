import streamlit as st
import os
from pathlib import Path
from typing import List, Dict, Any
import torch

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent))

from src.document_processor import DocumentProcessor
from src.rag_pipeline import RAGPipeline
from config.settings import PAGE_TITLE, PAGE_ICON, OLLAMA_BASE_URL


def init_session_state():
    """Initialize session state variables."""
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []


def main():
    """Main application function."""
    torch.classes.__path__ = []

    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    st.title("📚 RAG Chat Application")
    st.markdown("Chat with your PDF documents using AI")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # LLM Connection Status
        if st.session_state.rag_pipeline.check_llm_connection():
            st.success("✅ Ollama Connected")
            
            # Model Selection
            available_models = st.session_state.rag_pipeline.get_available_models()
            if available_models:
                selected_model = st.selectbox(
                    "Select Model",
                    available_models,
                    index=0 if available_models else None
                )
                if st.button("Set Model"):
                    st.session_state.rag_pipeline.set_llm_model(selected_model)
                    st.success(f"Model set to: {selected_model}")
            else:
                st.warning("No models found. Please pull a model first.")
        else:
            st.error("❌ Ollama Not Connected")
            st.markdown(f"Make sure Ollama is running on {OLLAMA_BASE_URL} and the model is available.")
        
        st.divider()
        
        # Knowledge Base Info
        st.header("📊 Knowledge Base")
        kb_info = st.session_state.rag_pipeline.get_knowledge_base_info()
        
        st.metric("Documents", kb_info.get('document_count', 0))
        st.metric("Unique Sources", kb_info.get('num_unique_sources', 0))
        
        if kb_info.get('unique_sources'):
            st.subheader("📄 Loaded Documents")
            for source in kb_info['unique_sources']:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(source)
                with col2:
                    if st.button("🗑️", key=f"delete_{source}"):
                        if st.session_state.rag_pipeline.remove_document_from_knowledge_base(source):
                            st.success(f"Removed {source}")
                            st.rerun()
                        else:
                            st.error("Failed to remove document")
        
        # Clear Knowledge Base
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.session_state.rag_pipeline.clear_knowledge_base():
                st.session_state.chat_history = []
                st.success("Knowledge base cleared!")
                st.rerun()
            else:
                st.error("Failed to clear knowledge base")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload PDF documents to chat with"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                process_documents(uploaded_files)
    
    with col2:
        st.header("💬 Chat Interface")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (role, message) in enumerate(st.session_state.chat_history):
                if role == "user":
                    st.chat_message("user").write(message)
                else:
                    st.chat_message("assistant").write(message)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.rag_pipeline.check_llm_connection():
                st.error("Please ensure Ollama is running and connected.")
                return
            
            kb_info = st.session_state.rag_pipeline.get_knowledge_base_info()
            if kb_info.get('document_count', 0) == 0:
                st.warning("Please upload and process some documents first.")
                return
            
            # Add user message to chat history
            st.session_state.chat_history.append(("user", prompt))
            
            # Display user message
            st.chat_message("user").write(prompt)
            
            # Generate and display response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                
                # Use streaming response
                full_response = ""
                for chunk in st.session_state.rag_pipeline.process_streaming_query(prompt):
                    full_response += chunk
                    response_placeholder.write(full_response + "▌")
                
                response_placeholder.write(full_response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append(("assistant", full_response))
            
            # Show sources (optional)
            if st.checkbox("Show Sources", key=f"sources_{len(st.session_state.chat_history)}"):
                show_sources(prompt)


def process_documents(uploaded_files):
    """Process uploaded PDF documents."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Process the PDF
            documents, metadata = st.session_state.doc_processor.process_pdf(uploaded_file)
            
            # Add to knowledge base
            if st.session_state.rag_pipeline.add_documents_to_knowledge_base(documents):
                st.success(f"✅ Successfully processed {uploaded_file.name}")
                st.json({
                    "Filename": metadata['filename'],
                    "File Size": f"{metadata['file_size']} bytes",
                    "Number of Chunks": metadata['num_chunks']
                })
            else:
                st.error(f"❌ Failed to process {uploaded_file.name}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("✅ All documents processed successfully!")
        
        # Update session state
        st.session_state.uploaded_files.extend([f.name for f in uploaded_files])
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
    
    finally:
        progress_bar.empty()
        status_text.empty()


def show_sources(query: str):
    """Show sources for the last query."""
    try:
        result = st.session_state.rag_pipeline.process_query(query)
        
        if result['sources']:
            st.subheader("📚 Sources")
            for source in result['sources']:
                with st.expander(f"📄 {source['name']} (Relevance: {source['relevance']:.2f})"):
                    st.text(source['chunk_info'])
        else:
            st.info("No sources found for this query.")
            
    except Exception as e:
        st.error(f"Error retrieving sources: {str(e)}")


if __name__ == "__main__":
    main()

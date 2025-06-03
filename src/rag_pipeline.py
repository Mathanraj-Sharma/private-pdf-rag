from typing import List, Dict, Any, Optional, Tuple
from src.vector_store import VectorStore
from src.llm_client import OllamaClient
from langchain.schema import Document
from config.settings import TOP_K_RESULTS, SIMILARITY_THRESHOLD


class RAGPipeline:
    """RAG (Retrieval-Augmented Generation) pipeline for query processing."""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm_client = OllamaClient()
    
    def process_query(self, query: str, k: int = TOP_K_RESULTS) -> Dict[str, Any]:
        """Process a user query through the complete RAG pipeline."""
        try:
            # Step 1: Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(query, k=k)
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                    "sources": [],
                    "context_used": "",
                    "success": True
                }
            
            # Step 2: Filter documents by similarity threshold
            filtered_docs = [
                doc for doc in relevant_docs 
                if doc.metadata.get('similarity_score', 0) >= SIMILARITY_THRESHOLD
            ]
            
            if not filtered_docs:
                return {
                    "answer": "I found some potentially relevant information, but it doesn't seem closely related to your question. Could you try rephrasing your query?",
                    "sources": [],
                    "context_used": "",
                    "success": True
                }
            
            # Step 3: Prepare context from retrieved documents
            context = self._prepare_context(filtered_docs)
            
            # Step 4: Generate response using LLM
            answer = self.llm_client.generate_response(query, context)
            
            # Step 5: Prepare sources information
            sources = self._extract_sources(filtered_docs)
            
            return {
                "answer": answer,
                "sources": sources,
                "context_used": context,
                "success": True,
                "num_sources": len(filtered_docs)
            }
            
        except Exception as e:
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "context_used": "",
                "success": False
            }
    
    def process_streaming_query(self, query: str, k: int = TOP_K_RESULTS):
        """Process query with streaming response."""
        try:
            # Always try to get relevant documents first
            relevant_docs = self.vector_store.similarity_search(query, k=k)
            
            # Check if we have good document matches
            filtered_docs = [
                doc for doc in relevant_docs 
                if doc.metadata.get('similarity_score', 0) >= SIMILARITY_THRESHOLD
            ]
            
            # Prepare context based on what we found
            if filtered_docs:
                # Good document matches - use document context
                context = self._prepare_context(filtered_docs)
                system_prompt = "Based on the provided context from the documents, please answer the user's question."
            elif relevant_docs and max(doc.metadata.get('similarity_score', 0) for doc in relevant_docs) > 0.2:
                # Some weak matches - use them but indicate uncertainty
                context = self._prepare_context(relevant_docs[:3])  # Use top 3 even if weak
                system_prompt = "Based on the potentially related information from the documents (though not a strong match), please try to answer the user's question. If the context isn't helpful, provide a general response."
            else:
                # No good matches - general response
                context = ""
                system_prompt = "Please provide a helpful response to the user's question."
            
            # Stream the response with appropriate context
            for chunk in self.llm_client.generate_streaming_response(
                query, 
                context, 
                system_prompt=system_prompt
            ):
                yield chunk
                
        except Exception as e:
            yield f"An error occurred: {str(e)}"
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context string from retrieved documents."""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            similarity = doc.metadata.get('similarity_score', 0)
            
            context_parts.append(
                f"[Source {i}: {source} (Relevance: {similarity:.2f})]\n"
                f"{doc.page_content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information from documents."""
        sources = []
        seen_sources = set()
        
        for doc in documents:
            source_name = doc.metadata.get('source', 'Unknown')
            
            if source_name not in seen_sources:
                sources.append({
                    'name': source_name,
                    'relevance': doc.metadata.get('similarity_score', 0),
                    'chunk_info': f"Chunk {doc.metadata.get('chunk_id', 0) + 1} of {doc.metadata.get('total_chunks', 1)}"
                })
                seen_sources.add(source_name)
        
        return sources
    
    def add_documents_to_knowledge_base(self, documents: List[Document]) -> bool:
        """Add new documents to the knowledge base."""
        return self.vector_store.add_documents(documents)
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get information about the current knowledge base."""
        collection_info = self.vector_store.get_collection_info()
        unique_sources = self.vector_store.get_unique_sources()
        
        return {
            **collection_info,
            "unique_sources": unique_sources,
            "num_unique_sources": len(unique_sources)
        }
    
    def remove_document_from_knowledge_base(self, source: str) -> bool:
        """Remove a document from the knowledge base."""
        return self.vector_store.delete_documents_by_source(source)
    
    def clear_knowledge_base(self) -> bool:
        """Clear the entire knowledge base."""
        return self.vector_store.clear_collection()
    
    def check_llm_connection(self) -> bool:
        """Check if LLM service is available."""
        return self.llm_client.check_connection()
    
    def get_available_models(self) -> List[str]:
        """Get list of available LLM models."""
        return self.llm_client.list_models()
    
    def set_llm_model(self, model_name: str):
        """Set the LLM model to use."""
        self.llm_client.set_model(model_name)

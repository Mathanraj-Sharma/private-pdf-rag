from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain.vectorstores import Chroma
from src.embeddings import EmbeddingHandler
from config.settings import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME


class VectorStore:
    """Manages ChromaDB vector store operations."""
    
    def __init__(self):
        self.embedding_handler = EmbeddingHandler()
        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        self.collection_name = CHROMA_COLLECTION_NAME
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get existing ChromaDB collection."""
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
        except chromadb.errors.NotFoundError:
            # Collection doesn't exist, create it
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # Explicitly set distance metric
                embedding_function=None  # Use your own embeddings
            )
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store."""
        try:
            # Prepare data for ChromaDB
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            embeddings = self.embedding_handler.embed_documents(documents)
            
            # Generate IDs
            ids = [f"{doc.metadata.get('file_hash', 'unknown')}_{doc.metadata.get('chunk_id', i)}" 
                   for i, doc in enumerate(documents)]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            return True
            
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        try:
            query_embedding = self.embedding_handler.embed_query(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            documents = []
            for i in range(len(results['documents'][0])):
                # For cosine distance: similarity = 1 - distance
                # For L2 distance: similarity = 1 / (1 + distance)
                distance = results['distances'][0][i]
                
                # Check what distance metric is being used
                if distance <= 2.0:  # Likely cosine distance (0-2 range)
                    similarity = 1 - (distance / 2)  # Normalize to 0-1
                else:  # Likely L2 distance
                    similarity = 1 / (1 + distance)
                
                doc = Document(
                    page_content=results['documents'][0][i],
                    metadata={
                        **results['metadatas'][0][i],
                        'similarity_score': similarity,
                        'raw_distance': distance  # Keep raw distance for debugging
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error performing similarity search: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "document_count": count,
                "embedding_dimension": self.embedding_handler.get_embedding_dimension()
            }
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            return {"name": self.collection_name, "document_count": 0}
    
    def delete_documents_by_source(self, source: str) -> bool:
        """Delete documents from a specific source file."""
        try:
            # Get all documents with the specified source
            results = self.collection.get(
                where={"source": source},
                include=["documents", "metadatas"]
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                return True
            return False
            
        except Exception as e:
            print(f"Error deleting documents: {str(e)}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(name=self.collection_name)
            self._initialize_collection()
            return True
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")
            return False
    
    def get_unique_sources(self) -> List[str]:
        """Get list of unique source files in the collection."""
        try:
            results = self.collection.get(include=["metadatas"])
            sources = set()
            for metadata in results['metadatas']:
                if 'source' in metadata:
                    sources.add(metadata['source'])
            return list(sources)
        except Exception as e:
            print(f"Error getting unique sources: {str(e)}")
            return []

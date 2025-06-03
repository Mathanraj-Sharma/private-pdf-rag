from typing import List
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL


class EmbeddingHandler:
    """Handles document embeddings using HuggingFace models."""
    
    def __init__(self):
        # Ensure same model for documents and queries
        self.model_name = EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
    
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts, normalize_embeddings=True)  # Add normalization
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode([query], normalize_embeddings=True)  # Add normalization
        return embedding[0].tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        # For all-MiniLM-L6-v2, the dimension is 384
        sample_embedding = self.embed_query("test")
        return len(sample_embedding)

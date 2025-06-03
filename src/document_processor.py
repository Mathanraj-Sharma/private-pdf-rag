import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, UPLOADS_DIR


class DocumentProcessor:
    """Handles PDF document processing, parsing, and chunking."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def save_uploaded_file(self, uploaded_file) -> Path:
        """Save uploaded file to disk and return file path."""
        file_path = UPLOADS_DIR / uploaded_file.name
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text content from PDF file."""
        text = ""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
        
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")
        
        return text
    
    def create_document_chunks(self, text: str, file_path: Path) -> List[Document]:
        """Split document text into chunks and create Document objects."""
        chunks = self.text_splitter.split_text(text)
        
        # Create document hash for tracking
        file_hash = self._generate_file_hash(file_path)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": str(file_path.name),
                    "file_path": str(file_path),
                    "file_hash": file_hash,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        
        return documents
    
    def process_pdf(self, uploaded_file) -> tuple[List[Document], Dict[str, Any]]:
        """Complete PDF processing pipeline."""
        # Save file
        file_path = self.save_uploaded_file(uploaded_file)
        
        # Extract text
        text = self.extract_text_from_pdf(file_path)
        
        # Create chunks
        documents = self.create_document_chunks(text, file_path)
        
        # Create metadata
        metadata = {
            "filename": uploaded_file.name,
            "file_size": uploaded_file.size,
            "num_chunks": len(documents),
            "file_hash": self._generate_file_hash(file_path)
        }
        
        return documents, metadata
    
    def _generate_file_hash(self, file_path: Path) -> str:
        """Generate MD5 hash of file for duplicate detection."""
        hash_md5 = hashlib.md5()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def get_processed_files(self) -> List[str]:
        """Get list of processed PDF files."""
        pdf_files = [f.name for f in UPLOADS_DIR.glob("*.pdf")]
        return pdf_files

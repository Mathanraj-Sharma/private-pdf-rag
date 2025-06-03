import os
import re
import shutil
import datetime
import PyPDF2
import platform
import psutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_directories(base_dir: Path, subdirs: List[str]) -> None:
    """Create necessary directories if they don't exist."""
    for subdir in subdirs:
        dir_path = base_dir / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {dir_path}")


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\-.,!?;:()\[\]{}\'\"@#$%^&*+=<>/\\|`~]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"


def validate_pdf_file(file_path: Path) -> bool:
    """Validate if file is a proper PDF."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return len(pdf_reader.pages) > 0
    except Exception as e:
        logger.error(f"PDF validation failed for {file_path}: {str(e)}")
        return False


def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing problematic characters."""
    import re
    # Replace spaces with underscores and remove special characters
    safe_name = re.sub(r'[^\w\-_.]', '_', filename)
    # Remove multiple consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    return safe_name


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text (rough approximation)."""
    # Rough estimation: 1 token ≈ 4 characters for English text
    return len(text) // 4


def chunk_text_by_tokens(text: str, max_tokens: int = 1000, overlap_tokens: int = 200) -> List[str]:
    """Chunk text by estimated token count."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for word in words:
        word_tokens = estimate_tokens(word)
        
        if current_tokens + word_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            overlap_words = current_chunk[-overlap_tokens//4:] if len(current_chunk) > overlap_tokens//4 else current_chunk
            current_chunk = overlap_words + [word]
            current_tokens = sum(estimate_tokens(w) for w in current_chunk)
        else:
            current_chunk.append(word)
            current_tokens += word_tokens
    
    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging."""
    
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
    }


def backup_database(source_dir: Path, backup_dir: Path) -> bool:
    """Create a backup of the database directory."""
    try:
        if source_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"chroma_backup_{timestamp}"
            shutil.copytree(source_dir, backup_path)
            logger.info(f"Database backup created: {backup_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Backup failed: {str(e)}")
        return False


def restore_database(backup_path: Path, target_dir: Path) -> bool:
    """Restore database from backup."""
    try:
        if backup_path.exists() and target_dir.parent.exists():
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(backup_path, target_dir)
            logger.info(f"Database restored from: {backup_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Restore failed: {str(e)}")
        return False


def health_check() -> Dict[str, bool]:
    """Perform basic health checks on the application."""
    from config.settings import UPLOADS_DIR, CHROMA_DB_DIR
    
    checks = {
        "uploads_dir_exists": UPLOADS_DIR.exists(),
        "uploads_dir_writable": os.access(UPLOADS_DIR, os.W_OK) if UPLOADS_DIR.exists() else False,
        "chroma_dir_exists": CHROMA_DB_DIR.exists(),
        "chroma_dir_writable": os.access(CHROMA_DB_DIR, os.W_OK) if CHROMA_DB_DIR.exists() else False,
    }
    
    return checks

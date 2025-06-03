# RAG Chat Application

A Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents using AI. Built with Streamlit, LangChain, ChromaDB, and Ollama.

## Features

- 📄 **PDF Document Processing**: Upload and process multiple PDF files
- 🔍 **Intelligent Search**: Vector-based semantic search through your documents
- 💬 **Interactive Chat**: Natural language conversations with your documents
- 🤖 **Local AI**: Uses Ollama for privacy-focused local AI inference
- 📊 **Document Management**: View, manage, and delete processed documents
- ⚡ **Streaming Responses**: Real-time streaming of AI responses
- 🎛️ **Model Selection**: Choose from available Ollama models

## Architecture

- **UI Framework**: Streamlit
- **LLM Host**: Ollama (local AI models)
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace Sentence Transformers
- **Document Processing**: LangChain + PyPDF2
- **Dependency Management**: UV

## Prerequisites

1. **Python 3.9+**
2. **Ollama**: Install and run Ollama locally
   ```bash
   # Install Ollama (visit https://ollama.ai for instructions)
   # Pull a model (e.g., llama2)
   ollama pull llama2
   ```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rag-chat-app
   ```

2. **Install UV** (if not already installed):
   ```bash
   pip install uv
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your preferences
   ```

5. **Start Ollama** (in a separate terminal):
   ```bash
   ollama serve
   ```

## Usage

1. **Start the application**:
   ```bash
   uv run streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload PDF documents**:
   - Use the file uploader in the left panel
   - Click "Process Documents" to add them to your knowledge base

4. **Start chatting**:
   - Ask questions about your documents in the chat interface
   - View sources and relevance scores for transparency

## Project Structure

```
rag-chat-app/
├── pyproject.toml              # UV dependency management
├── README.md                   # Project documentation
├── .env.example               # Environment variables template
├── app.py                     # Main Streamlit application
├── config/
│   └── settings.py           # Application settings
├── src/
│   ├── document_processor.py  # PDF processing and chunking
│   ├── vector_store.py       # ChromaDB operations
│   ├── embeddings.py         # Embedding functionality
│   ├── llm_client.py         # Ollama LLM client
│   └── rag_pipeline.py       # RAG query pipeline
├── data/
│   ├── uploads/              # Uploaded PDF files
│   └── chroma_db/            # ChromaDB storage
└── utils/
    └── helpers.py            # Helper functions
```

## Configuration

Key configuration options in `.env`:

- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)
- `OLLAMA_MODEL`: Default model to use (default: llama2)
- `CHUNK_SIZE`: Text chunk size for processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K_RESULTS`: Number of relevant chunks to retrieve (default: 5)
- `SIMILARITY_THRESHOLD`: Minimum similarity score for relevance (default: 0.7)

## Available Models

The application supports any model available in Ollama. Popular options:

- `llama2`: General-purpose model
- `codellama`: Code-focused model
- `mistral`: Fast and efficient model
- `llama2:13b`: Larger model for better quality

Pull models using:
```bash
ollama pull <model-name>
```

## Troubleshooting

### Common Issues

1. **"Ollama Not Connected"**:
   - Ensure Ollama is running: `ollama serve`
   - Check the URL in your `.env` file
   - Verify no firewall is blocking the connection

2. **"No models found"**:
   - Pull at least one model: `ollama pull llama2`
   - Restart the application

3. **PDF processing errors**:
   - Ensure PDFs are not password-protected
   - Check file size and format
   - Try with a different PDF

4. **Memory issues**:
   - Reduce `CHUNK_SIZE` in `.env`
   - Process fewer documents at once
   - Use a smaller embedding model

### Performance Tips

- Use smaller models for faster responses (e.g., `mistral`)
- Adjust chunk size based on your documents
- Clear the knowledge base periodically to free memory
- Process documents in batches for large collections

## Development

### Adding New Features

1. **Custom Document Processors**: Extend `document_processor.py`
2. **New Vector Stores**: Implement interface similar to `vector_store.py`
3. **Additional LLM Providers**: Create new client similar to `llm_client.py`

### Running Tests

```bash
uv run pytest tests/
```

### Code Quality

```bash
# Format code
uv run black src/ utils/ config/

# Check code quality
uv run flake8 src/ utils/ config/

# Type checking
uv run mypy src/ utils/ config/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ollama](https://ollama.ai) for local AI inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [LangChain](https://langchain.com) for RAG pipeline
- [Streamlit](https://streamlit.io) for the web interface

from typing import List, Dict, Any, Optional, Generator
import ollama
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL


class OllamaClient:
    """Client for interacting with Ollama LLM using the official ollama-python library."""
    
    def __init__(self):
        self.model = OLLAMA_MODEL
        # Set the base URL if different from default
        if OLLAMA_BASE_URL != "http://localhost:11434":
            ollama.Client.host = OLLAMA_BASE_URL
        self.client = ollama.Client()
    
    def check_connection(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            # Try to list models as a connection test
            self.client.list()
            return True
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            models_response = self.client.list()
            return [model['model'] for model in models_response.get('models', [])]
        except Exception:
            return []
    
    def generate_response(self, prompt: str, context: Optional[str] = None, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response from Ollama model."""
        try:
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            
            # Prepare the user message with context if provided
            if context and context.strip():
                # Document-based response
                user_content = f"""Context from documents:
                    {context}
                    User question: {prompt}
                    Please answer based on the context provided. If the context doesn't contain relevant information, say so and provide general guidance."""
            else:
                # General response
                user_content = f"""User question: {prompt}
                    Please provide a helpful and friendly response."""
            
            messages.append({
                'role': 'user',
                'content': user_content
            })
            
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options = {
                    'temperature': kwargs.get('temperature', 0.7),
                    'top_p': kwargs.get('top_p', 0.9),
                    'num_predict': kwargs.get('max_tokens', 512),
                    'num_ctx': 4096,
                    'num_thread': 8,
                }
            )
            
            return response['message']['content']
                
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_streaming_response(self, prompt: str, context: Optional[str] = None, system_prompt: Optional[str] = None, **kwargs) -> Generator[str, None, None]:
        """Generate streaming response from Ollama model."""
        try:
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({
                    'role': 'system',
                    'content': system_prompt
                })
            
            # Prepare the user message with context if provided
            if context and context.strip():
                # Document-based response
                user_content = f"""Context from documents:
                    {context}
                    User question: {prompt}
                    Please answer based on the context provided. If the context doesn't contain relevant information, say so and provide general guidance."""
            else:
                # General response
                user_content = f"""User question: {prompt}
                    Please provide a helpful and friendly response."""
            
            if len(user_content) > 4096:
                user_content = user_content[:4096] + '... [truncated]'
            
            messages.append({
                'role': 'user',
                'content': user_content
            })
            
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options = {
                    'temperature': kwargs.get('temperature', 0.7),
                    'top_p': kwargs.get('top_p', 0.9),
                    'num_predict': kwargs.get('max_tokens', 512),
                    'num_ctx': 4096,
                    'num_thread': -1,
                    'num_gpu': 10,
                }
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            yield f"Error generating response: {str(e)}"
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            self.client.pull(model_name)
            return True
        except Exception:
            return False
    
    def push_model(self, model_name: str) -> bool:
        """Push a model to Ollama registry."""
        try:
            self.client.push(model_name)
            return True
        except Exception:
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model from local Ollama instance."""
        try:
            self.client.delete(model_name)
            return True
        except Exception:
            return False
    
    def set_model(self, model_name: str):
        """Set the current model to use."""
        self.model = model_name
    
    def get_current_model(self) -> str:
        """Get the currently selected model."""
        return self.model
    
    def show_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about a model."""
        try:
            model = model_name or self.model
            return self.client.show(model)
        except Exception as e:
            return {"error": str(e)}
    
    def create_model(self, name: str, modelfile: str) -> bool:
        """Create a custom model from a Modelfile."""
        try:
            self.client.create(model=name, modelfile=modelfile)
            return True
        except Exception:
            return False
    
    def copy_model(self, source: str, destination: str) -> bool:
        """Copy a model with a new name."""
        try:
            self.client.copy(source=source, destination=destination)
            return True
        except Exception:
            return False
    
    def generate_embeddings(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embeddings for the given text."""
        try:
            model_name = model or self.model
            response = self.client.embeddings(model=model_name, prompt=text)
            return response.get('embedding', [])
        except Exception:
            return []
    
    def chat_with_history(self, messages: List[Dict[str, str]], **options) -> str:
        """Chat with conversation history."""
        try:
            default_options = {
                'temperature': 0.7,
                'top_p': 0.9,
                'num_predict': 1000
            }
            default_options.update(options)
            
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options=default_options
            )
            
            return response['message']['content']
        except Exception as e:
            return f"Error in chat: {str(e)}"
    
    def chat_with_history_streaming(self, messages: List[Dict[str, str]], **options) -> Generator[str, None, None]:
        """Chat with conversation history (streaming)."""
        try:
            default_options = {
                'temperature': 0.7,
                'top_p': 0.9,
                'num_predict': 1000
            }
            default_options.update(options)
            
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options=default_options
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            yield f"Error in chat: {str(e)}"

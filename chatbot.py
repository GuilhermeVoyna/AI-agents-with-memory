import os
from mem0 import Memory
from openai import OpenAI
from qdrant_client import QdrantClient
import dotenv

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_API_URL = os.getenv("QDRANT_API_URL")
user_id = "john"

class BaseChatbot:
    def __init__(self, collection_name, model_name, embedding_model,provider_name,client=None):
        self.config = self.create_config(collection_name, model_name, embedding_model,provider_name)
        self.memory = Memory.from_config(config_dict=self.config)
        self.messages = [{"role": "system", "content": "You are a personal AI Assistant."}]
        self.client = client

    def create_config(self, collection_name, model_name, embedding_model,provider_name):
        """Create a standard configuration for the chatbot."""
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": collection_name,
                    "client": QdrantClient(api_key=QDRANT_API_KEY,url=QDRANT_API_URL),  
                    
                },
            },
            "llm": {
                "provider": provider_name,
                "config": {
                    "model": model_name,
                    "temperature": 0.2,
                    "ollama_base_url": "http://localhost:11434",  # Check the base URL
                },
            },
            "embedder": {
                "provider": provider_name,
                "config": {
                    "model": embedding_model,
                    "ollama_base_url": "http://localhost:11434",
                    "embedding_dims": 1536,
                },
            },
            "version": "v1.1"
        }
    def add_message(self, message, user_id):
        """Add a message to memory."""
        try:
            result = self.memory.add(messages=message, user_id=user_id)
            return result
        except Exception as e:
            print(f"Error adding message to memory: {e}")
        return None
    def get_memories(self, user_id):
        """Retrieve all memories for a user."""
        try:
            memories = self.memory.get_all(user_id=user_id)
            return memories
        except Exception as e:
            print(f"Error getting memories: {e}")



    def get_memories(self, user_id):
        memories = self.memory.get_all(user_id=user_id)
        return [m['memory'] for m in memories['results']]

    def search_memories(self, query, user_id):
        memories = self.memory.search(query, user_id=user_id)
        return [m['memory'] for m in memories['results']]  


class OllamaChatbot(BaseChatbot):
    def __init__(self):
        super().__init__(
            collection_name="ye-ollama",
            model_name="llama2",
            embedding_model="nomic-embed-text:latest",
            provider_name="ollama",
        )


class GPTChatbot(BaseChatbot):


    def __init__(self):
        super().__init__(
            collection_name="ye-bot",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            provider_name="openai",
            client = OpenAI(api_key=OPENAI_API_KEY)

        )
    def ask_question(self, question, user_id):
        # Fetch previous related memories
        previous_memories = self.search_memories(question, user_id=user_id)
        prompt = question
        if previous_memories:
            prompt = f"User input: {question}\n Previous memories: {previous_memories}"
        self.messages.append({"role": "user", "content": prompt})

        # Generate response using GPT-4o-mini
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages
        )
        answer = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": answer})

        # Store the question in memory
        self.memory.add(question, user_id=user_id)
        return answer

import os
from mem0 import Memory
from mem0.configs.base import MemoryConfig
import dotenv
from openai import OpenAI
dotenv.load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

class BaseChatbot:
    def __init__(self, collection_name, model_name, embedding_model,provider_name):
        self.config = self.create_config(collection_name, model_name, embedding_model,provider_name)
        self.memory = Memory.from_config(config_dict=self.config)
        self.messages = [{"role": "system", "content": "You are a personal AI Assistant."}]

    def create_config(self, collection_name, model_name, embedding_model,provider_name):
        """Create a standard configuration for the chatbot."""
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": collection_name,
                    "host": "localhost",
                    "port": 6333,
                    "embedding_model_dims": 768,  # Adjust as needed
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
                "provider": "ollama",
                "config": {
                    "model": embedding_model,
                    "ollama_base_url": "http://localhost:11434",
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
        self.client = OpenAI()
        super().__init__(
            collection_name="ye-gpt",
            model_name="gpt-4o-mini",
            embedding_model="nomic-embed-text:latest",
            provider_name="openai",
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

def main():
    user_id = "john"
    gpt_chatbot = GPTChatbot()

    while True:
        question = input("Question: ")
        if question.lower() in ['q', 'exit']:
            print("Exiting...")
            break
        answer = gpt_chatbot.ask_question(question, user_id=user_id)
        print(f"Answer: {answer}")
        memories = gpt_chatbot.get_memories(user_id=user_id)
        print("Memories:")
        for memory in memories:
            print(f"- {memory}")
        print("-----")

    result = gpt_chatbot.add_message("hey my name is john and i like cakes",user_id="john");print(result)
    memories = gpt_chatbot.get_memories("john")
    print(memories)


# Example usage
if __name__ == "__main__":
    main()
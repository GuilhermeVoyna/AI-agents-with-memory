import os
import subprocess
from chatbot import GPTChatbot
import dotenv
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_API_URL = os.getenv("QDRANT_API_URL")
user_id = "john"


def main():

    gpt_chatbot = GPTChatbot()
    print("Starting API...")
    start_api()
    
    print("""{
"question": "fale do meu pth",
"user_id": "john"
}""" )
    
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

    memories = gpt_chatbot.get_memories("john")
    print(memories)

def start_api():
    """Inicia a API Flask em um processo separado."""
    
    subprocess.Popen(["python", "api.py"])

# Example usage
if __name__ == "__main__":
    main()
    
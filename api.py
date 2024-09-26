from flask import Flask, request, jsonify
from chatbot import GPTChatbot  # Importa a classe do chatbot

app = Flask(__name__)

# Inicializa o chatbot
gpt_chatbot = GPTChatbot()

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    user_id = data.get('user_id', 'john')  # Padrão para o ID do usuário 'john' se não for fornecido

    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        answer = gpt_chatbot.ask_question(question, user_id=user_id)
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/memories/<user_id>', methods=['GET'])
def get_memories(user_id):
    try:
        memories = gpt_chatbot.get_memories(user_id)
        return jsonify({"user_id": user_id, "memories": memories})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)


"""{
  "question": "fale do meu pth",
  "user_id": "john"
}""" 
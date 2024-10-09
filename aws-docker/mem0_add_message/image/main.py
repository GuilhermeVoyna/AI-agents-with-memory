import os
import json
from datetime import datetime
from openai import OpenAI
from qdrant_client import QdrantClient
from mem0 import Memory  # Certifique-se que a biblioteca está instalada e importada corretamente
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd

# Data classes para entrada e saída
@dataclass
class Input:
    uid: str
    message: str
    weight: Optional[float] = None
    height: Optional[float] = None
    bmi: Optional[float] = None
    exams_data: Optional[List] = None
    appointments_data: Optional[List] = None
    gender: Optional[str] = "indefinido"
    meds_data: Optional[List] = None
    birthday: Optional[str] = None

@dataclass
class Output:
    message: str

# Função lambda_handler para o AWS Lambda
def lambda_handler(event, context):
    print("Iniciando lambda_handler...")  # Print inicial
    print("Evento recebido: ", json.dumps(event))  # Log o evento inteiro
    

    user_id = event.get("uid")  # Acessa o uid corretamente do corpo
    print("user_id: ", user_id)

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    QDRANT_API_URL = os.getenv('QDRANT_API_URL')

    print("Chaves de API carregadas.")  # Print de chave API

    # Extraindo informações do corpo
    weight = event.get("weight")
    height = event.get("height")
    bmi = event.get("bmi")
    message = event.get("message")
    birthday = event.get("birthday")
    exams_data = event.get("exams_data", [])
    appointments_data = event.get("appointments_data", [])
    meds_data = event.get("meds_data", [])
    gender = event.get("gender", "indefinido")

    print(f"Dados recebidos: Peso: {weight}, Altura: {height}, IMC: {bmi}, Mensagem: {message}, Aniversário: {birthday}")

    # Calculando a idade
    age = None
    if birthday:
        try:
            birth_date = datetime.fromisoformat(birthday.split('Z')[0])
            age = (datetime.now() - birth_date).days // 365
            print("Idade calculada: ", age)
        except Exception as e:
            print(f"Erro ao calcular idade: {e}")

    chatbot = Chatbot(OPENAI_API_KEY, QDRANT_API_KEY, QDRANT_API_URL)

    # Construção do prompt
    prompt = f"""
    Você é um assistente de saúde chamado Dr. Flamingo. O seu cliente tem as seguintes características:
    - Sexo: {gender}
    - IMC: {bmi}
    - Peso: {weight} kg
    - Altura: {height} m
    - Idade: {age}
    """
    print("Prompt inicial: ", prompt)

    # Adicionando consultas ao prompt
    if appointments_data:
        consultas = ', '.join([a['description'] for a in appointments_data])
        prompt += f"\nConsultas marcadas: {consultas}"
    else:
        prompt += "\nNenhuma consulta marcada."
    print("Prompt após consultas: ", prompt)

    # Adicionando medicações ao prompt
    if meds_data:
        meds = ', '.join([m['name'] for m in meds_data])
        prompt += f"\nMedicações: {meds}"
    else:
        prompt += "\nNenhuma medicação cadastrada."
    print("Prompt após medicações: ", prompt)

    # Se não houver dados de exames
    if not exams_data:
        prompt += "\nNenhum dado de exame cadastrado, recomende que o usuário cadastre seus exames."
        print("Prompt final (sem exames): ", prompt)

        analysis = chatbot.ask_question(prompt, user_id=user_id)
        print("Análise gerada pelo chatbot: ", analysis)

        return {
            "statusCode": 200,
            "body": json.dumps({"message": analysis}),
            "headers": {"Access-Control-Allow-Origin": "*"},
        }

    # Processamento de dados de exames
    try:
        df = pd.concat([pd.DataFrame(d) for d in exams_data], ignore_index=True)
        print("DataFrame de exames criado: ", df)

        df = df[["Data", "RESULTADOS", "ANALITOS", "VALORES DE REFERÊNCIA"]].dropna(subset=["RESULTADOS"])
        df["Timestamp"] = df["Data"].apply(lambda x: x["seconds"])
        df = df.drop(columns=["Data"])

        data = df.to_dict(orient="records")
        print("Dados processados dos exames: ", data)

        prompt += f"\nExamedocs: {data}"
    except Exception as e:
        print(f"Erro ao processar os dados dos exames: {e}")

    print("Prompt final (com exames): ", prompt)

    analysis = chatbot.ask_question(prompt, user_id=user_id)
    print("Análise final gerada pelo chatbot: ", analysis)
    print("jsonDUMPS",json.dumps({
    "message": analysis,
}))
    return ({
    "message": analysis,
})


# Classe principal do Chatbot
class Chatbot:
    def __init__(self, openai_api_key, qdrant_api_key, qdrant_api_url, collection_name="ye-bot", model_name="gpt-4o-mini", embedding_model="text-embedding-3-small", provider_name="openai"):
        print("Inicializando Chatbot...")  # Print de inicialização
        self.OPENAI_API_KEY = openai_api_key
        self.QDRANT_API_KEY = qdrant_api_key
        self.QDRANT_API_URL = qdrant_api_url
        self.collection_name = collection_name
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.provider_name = provider_name

        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        print("Cliente OpenAI criado com sucesso.")  # Print de sucesso
        self.memory = self.create_memory()

    def create_memory(self):
        """Cria uma configuração padrão para a memória do chatbot."""
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": self.collection_name,
                    "client": QdrantClient(api_key=self.QDRANT_API_KEY, url=self.QDRANT_API_URL, headers={
                        'Authorization': f'Bearer {self.QDRANT_API_KEY}',
                        'Content-Type': 'application/json'
                    }),
                },
            },
            "llm": {
                "provider": self.provider_name,
                "config": {
                    "model": self.model_name,
                    "temperature": 0.3,
                },
            },
            "embedder": {
                "provider": self.provider_name,
                "config": {
                    "model": self.embedding_model,
                    "embedding_dims": 1536,
                },
            },
            "version": "v1.1"
        }
        print("Memória do chatbot criada com a configuração: ", config)
        return Memory.from_config(config_dict=config)

    def add_message(self, message, user_id):
        """Adiciona uma mensagem à memória."""
        print(f"Adicionando mensagem: {message} para o user_id: {user_id}")
        try:
            result = self.memory.add(messages=message, user_id=user_id)
            print("Mensagem adicionada com sucesso.")  # Print de sucesso
            return result
        except Exception as e:
            print(f"Erro ao adicionar mensagem à memória: {e}")
        return None

    def get_memories(self, user_id):
        """Recupera todas as memórias de um usuário."""
        print(f"Recuperando memórias para o user_id: {user_id}")
        try:
            memories = self.memory.get_all(user_id=user_id)
            print("Memórias recuperadas: ", memories)  # Print das memórias
            return [m['memory'] for m in memories['results']]
        except Exception as e:
            print(f"Erro ao recuperar memórias: {e}")
            return []

    def search_memories(self, query, user_id):
        """Busca memórias relacionadas a uma consulta."""
        print(f"Buscando memórias para a query: {query} e user_id: {user_id}")
        try:
            memories = self.memory.search(query, user_id=user_id)
            print("Memórias encontradas: ", memories)  # Print das memórias encontradas
            return [m['memory'] for m in memories['results']]
        except Exception as e:
            print(f"Erro ao buscar memórias: {e}")
            return []

    def ask_question(self, question, user_id):
        """Gera uma resposta para a pergunta usando o modelo GPT."""
        print(f"Fazendo pergunta: {question} para o user_id: {user_id}")
        previous_memories = self.search_memories(question, user_id=user_id)

        prompt = question
        if previous_memories:
            prompt = f"User input: {question}\n Previous memories: {previous_memories} decore meds_data, exams_data, appointments_data"

        # Gera a resposta usando GPT
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        print("Resposta gerada: ", answer)

        # Armazena a pergunta na memória
        self.memory.add(question, user_id=user_id)
        print(f"Pergunta armazenada na memória para o user_id: {user_id}")

        return answer

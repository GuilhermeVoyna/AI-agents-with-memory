import boto3
import os
import json
from datetime import datetime , timezone
from openai import OpenAI
from qdrant_client import QdrantClient
from mem0 import Memory
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import threading  # Importar threading para execução assíncrona

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
    prev_messages: Optional[List] = None

@dataclass
class Output:
    message: str

def should_add_memory(date=None):
    print("Checking if memory should be updated...")
    if date is None:
        return True
    gpt_date = date
    gpt_date_parsed = datetime.fromisoformat(gpt_date)
    print("GPT Date in UTC:", gpt_date_parsed)
    
    # Obtenção da hora atual em UTC usando timezone correto
    now = datetime.now(timezone.utc)
    print("Now in UTC:", now)
    
    gpt_date_utc = gpt_date_parsed.astimezone(timezone.utc)
    time_diff = now - gpt_date_utc
    print("Time difference:", time_diff)

    if time_diff.total_seconds() > 24 * 3600:
        update_memory = True
    else:
        update_memory = False
    return update_memory


# Função lambda_handler para o AWS Lambda
def lambda_handler(event, context):
    print("Iniciando lambda_handler...")
    print("Evento recebido:", json.dumps(event)[:50])

    user_id = event.get("uid")
    message = event.get("message")
    user_name= event.get("uname")
    prev_messages = event.get("messages", [])
    prev_messages = prev_messages[-5:]
    print("prev_messages:", prev_messages)
    print("user_id:", user_id)
    exams_data = event.get("exams_data", [])
    # Carregamento de chaves de API
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    QDRANT_API_URL = os.getenv('QDRANT_API_URL')
    
    # Inicialização do chatbot
    chatbot = Chatbot(OPENAI_API_KEY, QDRANT_API_KEY, QDRANT_API_URL)
    latest_memory = chatbot.get_latest_memory(user_id, limit=1)
    
    if latest_memory is None:
        update = True
    elif "created_at" in latest_memory:
        update = should_add_memory(latest_memory["created_at"])
    else:
        update = True

    print("update:", update)
    
    if update:
        weight = event.get("weight")
        height = event.get("height")
        bmi = event.get("bmi")
        birthday = event.get("birthday")
        appointments_data = event.get("appointments_data", [])
        meds_data = event.get("meds_data", [])
        gender = event.get("gender", "indefinido")

        print(f"Dados recebidos: Peso: {weight}, Altura: {height}, IMC: {bmi}, Mensagem: {message}, Aniversário: {birthday}")

        # Cálculo de idade
        age = None
        if birthday:
            try:
                birth_date = datetime.fromisoformat(birthday.split('Z')[0])
                age = (datetime.now() - birth_date).days // 365
                print("Idade calculada:", age)
            except Exception as e:
                print(f"Erro ao calcular idade: {e}")

        # Construção do prompt
        prompt = f"""
        Você é um assistente de saúde chamado Dr. Flamingo. O seu cliente chamado {user_name} tem as seguintes características:
        - Sexo: {gender}
        - Peso: {weight} kg
        - Idade: {age or 'Indefinida'}
        """

        if appointments_data:
            consultas = ', '.join([a['description'] for a in appointments_data])
            prompt += f"\nConsultas marcadas: {consultas}"
        else:
            prompt += "\nNenhuma consulta marcada."

        if meds_data:
            meds = ', '.join([m['name'] for m in meds_data])
            prompt += f"\nMedicações: {meds}"
        else:
            prompt += "\nNenhuma medicação cadastrada."

        if not exams_data:
            prompt += "\nNenhum dado de exame cadastrado, recomende que o usuário cadastre seus exames."

        print("Prompt final:", prompt[:50])

        invoke_lambda_add_message(prompt, user_id, lambdaname="mem0_add_message")

    # Geração da análise imediatamente
    analysis = chatbot.ask_question(message, user_id=user_id, exams_data=exams_data, prev_messages=prev_messages,user_name=user_name)
    print("Análise final gerada pelo chatbot:", analysis)

    # Retorna a resposta imediatamente
    response = {
        "message": analysis
    }

    # Executa o `add_message` em uma lambda 
    invoke_lambda_add_message(analysis, user_id)
    
    print("Resposta final:", response)
    return response

def invoke_lambda_add_message(message, user_id,lambdaname="mem0_add_message"):
    """Invoca outra função Lambda."""
    client = boto3.client('lambda')

    payload = {
        'message': message,
        'user_id': user_id
    }

    try:
        response = client.invoke(
            FunctionName=lambdaname,
            InvocationType='Event',
            Payload=json.dumps(payload)
        )
        print(f"Invocação assíncrona da Lambda bem-sucedida. Resposta: {response}")
    except Exception as e:
        print(f"Erro ao invocar a Lambda: {e}")

class Chatbot:
    def __init__(self, openai_api_key, qdrant_api_key, qdrant_api_url, collection_name="ye-bot", model_name="gpt-4o-mini", embedding_model="text-embedding-3-small", provider_name="openai"):
        print("Inicializando Chatbot...")  # Mensagem de inicialização
        self.OPENAI_API_KEY = openai_api_key
        self.QDRANT_API_KEY = qdrant_api_key
        self.QDRANT_API_URL = qdrant_api_url
        self.collection_name = collection_name
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.provider_name = provider_name

        # Inicializa cliente da API OpenAI
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        print("Cliente OpenAI criado com sucesso.")  # Mensagem de sucesso
        self.memory = self.create_memory()

        # Armazena mensagens do histórico de conversa
        self.messages = []

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
        print(f"Adicionando mensagem...")
        try:
            result = self.memory.add(messages=message, user_id=user_id)
            print("Mensagem adicionada com sucesso.")
            return result
        except Exception as e:
            print(f"Erro ao adicionar mensagem à memória: {e}")
        return None

    def get_memories(self, user_id):
        """Recupera todas as memórias de um usuário."""
        print(f"Recuperando memórias para o user_id: {user_id}")
        try:
            memories = self.memory.get_all(user_id=user_id)
            print("Memórias recuperadas: ", memories)
            return [m['memory'] for m in memories['results']]
        except Exception as e:
            print(f"Erro ao recuperar memórias: {e}")
            return []

    def search_memories(self, query, user_id,limit=5):
        """Busca memórias relacionadas a uma consulta."""
        print(f"Buscando memórias para a query: {query} e user_id: {user_id}")
        try:
            memories = self.memory.search(query, user_id=user_id, limit=limit)
            return [m['memory'] for m in memories['results']]  
        except Exception as e:
            print(f"Erro ao buscar memórias: {e}")
            return []
    
    def get_latest_memory(self, user_id, limit=1):
        """Recupera a memória mais recente de um usuário."""
        print(f"Recuperando a memória mais recente para o user_id: {user_id}")
        try:
            memories = self.memory.get_all(user_id=user_id, limit=limit)
            sorted_memories = sorted(
                memories['results'], 
                key=lambda m: m['created_at'] if 'created_at' in m else m['updated_at'], 
                reverse=True
            )
            latest_memory = sorted_memories[0] if sorted_memories else None
            print("Memória mais recente: ", latest_memory)
            return latest_memory
        except Exception as e:
            print(f"Erro ao recuperar a memória mais recente: {e}")
            return None
    def extra_data(self,question,dados=[]):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Você é um robô que apenas responde SIM ou NÃO. Caso tenha duvidas responda SIM"},
                {"role": "user", "content": f"Você acredita que a próxima resposta da LLM vai precisar de dados médicos EXTRAS para responder a pergunta sendo que possue esses dados {dados}?\nPergunta: {question}\n caso nao tenha dados responda sim"}
            ],
            max_tokens=50,
            temperature=0.1,
        )
        print(dados)
        answer = response.choices[0].message.content
        if "SIM" in answer:
            print("SIM")
            return True
        if "NÃO" in answer:
            print("NÃO")
            return False
        else:
            print("Error: resposta não esperada")
            return False

    def ask_question(self, question, user_id,exams_data,prev_messages=[],user_name="Usuário"):
        """Gera uma resposta para a pergunta usando o modelo GPT."""
        print(f"Fazendo pergunta: {question} para o user_id: {user_id}")
        previous_memories = self.search_memories(question, user_id=user_id,limit=5)
        prompt = "Seu nome é Dr Flamingo, você é um assistente de saúde.\n"

        if previous_memories:
            prompt = prompt + f" Previous memories: {previous_memories}"
        if self.extra_data(question,dados=[previous_memories+prev_messages]):
            try:
                df = pd.concat([pd.DataFrame(d) for d in exams_data], ignore_index=True)
                df = df[["Data", "RESULTADOS", "ANALITOS", "VALORES DE REFERÊNCIA"]].dropna(subset=["RESULTADOS"])
                df["Timestamp"] = df["Data"].apply(lambda x: x["seconds"])
                df = df.drop(columns=["Data"])
                data = df.to_dict(orient="records")
                prompt = f"Dados médicos {data}" + prompt
            except Exception as e:
                print(f"Erro ao processar os dados dos exames: {e}")
                prompt += "\nErro ao processar os dados dos exames, recomende que o usuário contate um adiministrador."         

        # Gera a resposta usando GPT
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
            *prev_messages,
            {"role": "system", "content": "SEU NOME É Dr. Flamingo e responda perguntas medicas. O nome do seu cliente é "+user_name},
            {"role": "assistant", "content": prompt},
            {
                "role": "user",
                "content": f"Responda em português: {question}",
            },
        ],
        )
        answer = response.choices[0].message.content
        print("Resposta gerada: ", answer)

        # Armazena a pergunta na memória
        print(f"Pergunta armazenada na memória para o user_id: {user_id}")

        return answer

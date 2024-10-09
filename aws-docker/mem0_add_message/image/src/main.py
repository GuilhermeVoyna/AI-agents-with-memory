import json
import os
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

def lambda_handler(event, context):
    
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    QDRANT_API_URL = os.getenv('QDRANT_API_URL')
    # Inicialização do chatbot
    chatbot = Chatbot(OPENAI_API_KEY, QDRANT_API_KEY, QDRANT_API_URL)
    print("Chatbot inicializado com sucesso.")  # Mensagem de sucesso
    print("Evento recebido: ", json.dumps(event))  # Log do evento recebido
    try:
        # Tentando parsear o corpo do evento (caso seja uma requisição HTTP com body JSON)
        if 'body' in event:
            body = json.loads(event['body'])
            user_id = body.get('user_id')
            message = body.get('message')
        else:
            # Caso não haja body, tenta pegar diretamente dos parâmetros
            user_id = event.get('user_id')
            message = event.get('message')
        # Adicionando mensagem de forma assíncrona à memória
        prompt = "Voce consegue decorar apenas dados médico e o nome do seu analito mas so consegue decorar APENAS os dados medicos"
        if user_id == "public":
            prompt = "Voce consegue decorar apenas dicas que podem ser compartilhadas com o publico"
        chatbot.add_message(message, user_id,prompt)

        # Retornando uma resposta de sucesso
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Mensagem adicionada com sucesso!',
                'user_id': user_id
            })
        }
    except Exception as e:
        # Retorno em caso de erro
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

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

    def add_message(self, message, user_id,prompt):
        """Adiciona uma mensagem à memória."""
        print(f"Adicionando mensagem...")
        try:
            result = self.memory.add(messages=message, user_id=user_id,prompt=prompt)
            print("Mensagem adicionada com sucesso.")
            return result
        except Exception as e:
            print(f"Erro ao adicionar mensagem à memória: {e}")
        return None


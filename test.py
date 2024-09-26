import os
from mem0 import Memory
from mem0.configs.base import MemoryConfig

# Configuração do MemoryConfig para utilizar a versão 'v1.1'
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test",
            "host": "localhost",
            "port": 6333,
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1:latest",
            "temperature": 0,
            "max_tokens": 8000,
            "ollama_base_url": "http://localhost:11434",  # Ensure this URL is correct
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",
            # Alternatively, you can use "snowflake-arctic-embed:latest"
            "ollama_base_url": "http://localhost:11434",
        },
        
    },
    "version": "v1.1"
}

# Criando a instância da classe Memory com a configuração
m = Memory.from_config(config_dict=config)

# Exemplo de uso para adicionar e buscar memórias
result = m.add("I'm visiting Paris", user_id="john")
print(result)

memories = m.get_all(user_id="john")
print(memories)

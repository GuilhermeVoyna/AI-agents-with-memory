from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from dotenv import load_dotenv
load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_API_URL = os.getenv("QDRANT_API_URL")
print(QDRANT_API_URL)

def setup():
    qdrant_client = QdrantClient(
        url=QDRANT_API_URL, 
        api_key=QDRANT_API_KEY,
    )


    collection_config = models.VectorParams(
        size=1536,
        distance=models.Distance.COSINE
    )

    qdrant_client.create_collection(
        collection_name="ye-bot",#collection_name,
        vectors_config=collection_config
    )

if __name__ == "__main__":
    #input y or n
    input=input("Are you sure you want to create a new collection? (y/n): ")
    if input == "y":
        setup()
    else:
        print("Collection creation aborted.")
        exit()
import os
from dotenv import load_dotenv

load_dotenv()
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

pc.create_index(
    name=os.environ["DOC_INDEX_NAME"],
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1",
    ),
)

print("Index created successfully!")

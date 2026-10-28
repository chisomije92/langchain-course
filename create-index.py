import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

pc.create_index(
    # name=os.environ["INDEX_NAME"],
    name="medium-emb-v2",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1",
    ),
)
# print(pc.describe_index(os.environ["INDEX_NAME"]))
print("Index created successfully!")

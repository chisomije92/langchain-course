import os
from pathlib import Path
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).parent
FILE_PATH = str(BASE_DIR / "blog-text.txt")


def main():
    print("Starting ingestion process...")
    loader = UnstructuredLoader(
        # file_path="/Users/Chisom/Documents/AI Projects/langchain-course/blog-text.txt",
        file_path=FILE_PATH,
        chunking_strategy="basic",
        max_characters=1000000,
    )
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(api_key=SecretStr(os.environ["OPENAI_API_KEY"]))

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name="medium-emb-v2",
        # index_name=os.environ["INDEX_NAME"]
    )
    print("finish")


if __name__ == "__main__":
    main()

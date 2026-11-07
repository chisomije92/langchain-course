import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def load_webpage(url: str) -> list[Document]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unnecessary HTML elements
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    content = soup.get_text(separator=" ", strip=True)

    return [
        Document(
            page_content=content,
            metadata={"source": url},
        )
    ]


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Step 1: Load documents
docs_list = []

for url in urls:
    docs_list.extend(load_webpage(url))


# Step 2: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0,
)

doc_splits = text_splitter.split_documents(docs_list)


# Step 3: Initialize embeddings
embeddings = OpenAIEmbeddings()


# Step 4: Create and persist vector store
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
    persist_directory="./.chroma",
)


# Step 5: Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# Step 6: Test retrieval
results = retriever.invoke("What are AI agents?")

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("---")

from src.helper import load_pdf_file, text_split, download_huggingface_embeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

INDEX_NAME = "mental-health-chatbot"

# Load documents
documents = load_pdf_file(data="Data/")
text_chunks = text_split(documents)

# Embeddings (384 DIM)
embeddings = download_huggingface_embeddings()

pc = PineconeClient(api_key=PINECONE_API_KEY)

# 🚨 ALWAYS CREATE WITH 384
pc.create_index(
    name=INDEX_NAME,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Upload vectors
Pinecone.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("✅ Pinecone index created with 384 dims and documents uploaded")
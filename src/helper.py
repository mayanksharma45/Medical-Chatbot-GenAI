from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



# Split the Data into Text Chunks

def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=25)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks



# Download the Embeddings from google genai

def download_google_genai_embeddings():
    embeddings=GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    return embeddings
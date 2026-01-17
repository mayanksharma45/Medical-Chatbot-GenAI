from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



# Split the Data into Text Chunks

def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks



def download_huggingface_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

import streamlit as st
import os
from dotenv import load_dotenv

from src.helper import download_huggingface_embeddings
from src.prompt import system_prompt

from langchain_community.vectorstores import Pinecone
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="🩺 Medical Chatbot - GenAI",
    page_icon="🧠",
    layout="centered"
)

st.title("🩺 Medical Chatbot (GenAI)")
st.warning(
    "⚠️ This chatbot is for **educational purposes only** and does NOT provide medical advice. "
    "Please consult a qualified healthcare professional for medical concerns."
)

# -----------------------------------
# Load Environment Variables
# -----------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
    st.error("Missing API keys. Please set them in Streamlit Secrets.")
    st.stop()

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# -----------------------------------
# Cache Heavy Resources
# -----------------------------------
@st.cache_resource
def load_rag_pipeline():
    # Embeddings
    embeddings = download_huggingface_embeddings()

    # Pinecone Vector Store
    index_name = "mental-health-chatbot"
    vectorstore = Pinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
        # max_output_tokens: 200
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


rag_chain = load_rag_pipeline()


# -----------------------------------
# Chat UI
# -----------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a medical or mental health related question...")

if user_input:
    # Save user message
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Thinking... 🤔"):
        response = rag_chain.invoke({"input": user_input})
        answer = response["answer"]

    # Save bot response
    st.session_state.chat_history.append(("assistant", answer))


# -----------------------------------
# Display Chat History
# -----------------------------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

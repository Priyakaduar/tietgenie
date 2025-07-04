import os
import tempfile
import base64
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_together import ChatTogether

# ---------------- SETUP ----------------
load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")
st.set_page_config(page_title="Tiet-Genie 🤖", layout="wide")

# ✅ Background image with visible overlay
def set_bg_with_overlay(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .main > div:has(.block-container) {{
        background: url("data:image/jpg;base64,{b64}") no-repeat center center fixed;
        background-size: cover;
        position: relative;
    }}
    .main > div:has(.block-container)::before {{
        content: "";
        background-color: rgba(255, 255, 255, 0.82);
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        z-index: 0;
    }}
    .block-container {{
        position: relative;
        z-index: 1;
    }}
    .stChatMessageContent, .stMarkdown {{
        color: #111 !important;
        font-weight: 500;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg_with_overlay("thaparbg.jpg")  # Your local background image

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("TIETlogo.png", width=120)
    st.markdown("## 🤖 Tiet-Genie")
    st.markdown("How can I assist you today? 😊")
    uploaded = st.file_uploader("📎 Attach additional PDFs", type="pdf", accept_multiple_files=True)

# ---------------- PRELOADED PDFs ----------------
@st.cache_resource(show_spinner="Loading TIET manuals...")
def load_preloaded():
    docs = []
    for path in ["rules.pdf"]:
        docs.extend(PyPDFLoader(path).load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embed)

vs = load_preloaded()

# ---------------- USER PDF Handling ----------------
if uploaded:
    docs = []
    for f in uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            docs.extend(PyPDFLoader(tmp.name).load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    user_vs = FAISS.from_documents(chunks, embed)
    vs.merge_from(user_vs)

# ---------------- RAG + LLM ----------------
retriever = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)

llm = ChatTogether(
    model="deepseek-ai/DeepSeek-V3",
    temperature=0.2,
    together_api_key=together_api_key
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ---------------- CHAT STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "greeted" not in st.session_state:
    st.session_state.greeted = False

# ---------------- GREETING ----------------
if not st.session_state.greeted and not st.session_state.chat_history:
    st.markdown("""
    <h2 style='text-align: center; color: #8B0000; font-weight: bold; margin-top: 30px;'>
        👋 Hello TIETian! How can I help you today?
    </h2>
    """, unsafe_allow_html=True)

# ---------------- CHAT UI ----------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["message"])

user_input = st.chat_input("Ask a question...")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    st.session_state.greeted = True

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = qa_chain({"query": user_input})
                response = result["result"]
            except Exception as e:
                response = f"⚠ Error: {str(e)}"
        st.write(response)
        st.session_state.chat_history.append({"role": "assistant", "message": response})


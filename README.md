# tietgenie
# 🤖 Tiet-Genie: University Manual Chatbot (RAG + LangChain + Streamlit)

Tiet-Genie is a chatbot that answers queries from Thapar Institute's HR and academic manuals using **RAG (Retrieval-Augmented Generation)**. It’s powered by **LangChain**, **Hugging Face embeddings**, and deployed on **Streamlit Cloud**. Users can also upload their own PDFs for dynamic question answering.

---

## Features

-  Preloaded with university manuals (`rules.pdf`)
-  Upload your own PDFs for custom queries
-  Semantic retrieval using **FAISS** + Hugging Face
-  Deployed via **Streamlit Cloud** with ChatGPT-style UI
-  Uses **DeepSeek-V3** via ChatTogether API for LLM responses

---

  ## Tech Stack

- **LLM:** DeepSeek-V3 (ChatTogether API)  
- **Embedding Model:** `all-MiniLM-L6-v2` (Hugging Face)  
- **RAG Framework:** LangChain  
- **Vector Store:** FAISS  
- **UI & Deployment:** Streamlit Cloud  
- **Language:** Python  

---

## Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/Priyakaduar/tiet-genie.git
cd tiet-genie

### Install DEpendencies
pip install -r requirements.txt

### Set Your Environment Key
TOGETHER_API_KEY=your_chat_together_api_key

### RUn the App
streamlit run app.py

Live Demo: https://tietgenie-lhetbqhsj7igti9ghfx9nc.streamlit.app/





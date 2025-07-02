import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_together import ChatTogether

print("🔧 Script is running...")

# ✅ Step 1: Check API key
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("🚫 TOGETHER_API_KEY is not set!")
print("🔐 API key detected.")

# ✅ Step 2: Load PDF
pdf_path = "rules.pdf"  # 🔁 Change this to your actual PDF name if needed
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"❌ PDF not found at path: {pdf_path}")

loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"📄 Loaded {len(documents)} page(s) from PDF.")

# ✅ Step 3: Text splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)
print(f"✂️ Split into {len(split_docs)} chunks.")

# ✅ Step 4: Embeddings + FAISS vector store
print("🧠 Embedding and indexing...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ✅ Step 5: Load Together LLM
print("🤖 Loading Together LLM...")
llm = ChatTogether(model="deepseek-ai/DeepSeek-V3", temperature=0.2)

# ✅ Step 6: Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ✅ Step 7: Ask questions
while True:
    query = input("\n❓ Ask a question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        print("👋 Exiting.")
        break

    print("💬 Thinking...")
    try:
        result = qa_chain(query)
        clean_answer = result["result"].replace("**", "")
        print("\n✅ Answer:\n", clean_answer)


        print("\n📚 Source snippet(s):")
        for i, doc in enumerate(result["source_documents"]):
            print(f"- Snippet {i+1}:", doc.page_content[:300], "...\n")
    except Exception as e:
        print("❌ Error during QA:", e)


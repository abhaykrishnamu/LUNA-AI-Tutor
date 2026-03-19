# --- 1. CHROMADB SQLITE FIX ---
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

# --- 2. PAGE CONFIG ---
st.set_page_config(page_title="LUNA AI - C Tutor", page_icon="🌙", layout="wide")

# --- 3. API SETUP ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("API key not found. Please add GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- 4. LOAD KNOWLEDGE BASE (ROOT FOLDER) ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base(_api_key):
    # "." tells Python to look in the same folder where main.py is located
    folder = "." 
    db_dir = "./chroma_db_c"

    # Load only PDF files from the root directory
    loader = PyPDFDirectoryLoader(folder)
    docs = loader.load()

    if not docs:
        return None, 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=_api_key,
        task_type="retrieval_document"
    )

    if os.path.exists(db_dir):
        try: shutil.rmtree(db_dir)
        except: pass

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )

    return vector_db, len(docs)

# --- 5. UI SETUP ---
st.title("🌙 LUNA AI: C Programming Tutor")
st.caption("KTU Engineering | AI & ML Department | Jai Bharath College")

with st.spinner("LUNA is reading your KTU notes from the main folder..."):
    vector_db, doc_count = load_knowledge_base(api_key)

# --- 6. SIDEBAR ---
with st.sidebar:
    st.header("🌙 LUNA Settings")
    if vector_db:
        st.success(f"📚 {doc_count} pages of C notes loaded")
    else:
        st.warning("No PDFs found in the root folder!")

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"Developed by Abhay Krishna MU")

# --- 7. CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a C programming question...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").markdown(user_query)

    with st.spinner("LUNA is thinking..."):
        system_prompt = "You are LUNA, an expert C tutor for KTU students. Use C code blocks and LaTeX."
        
        if vector_db:
            docs = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            full_prompt = f"{system_prompt}\n\nContext from Notes:\n{context}\n\nQuestion: {user_query}"
        else:
            full_prompt = f"{system_prompt}\n\nQuestion: {user_query}"
            
        response = model.generate_content(full_prompt)
        answer = response.text

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

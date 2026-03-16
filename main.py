# --- 1. CHROMADB COMPATIBILITY FIX (MUST BE AT THE VERY TOP) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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
st.set_page_config(
    page_title="LUNA C-Tutor",
    page_icon="💻",
    layout="wide"
)

st.title("💻 LUNA AI: C Programming Tutor")
st.caption("KTU Engineering student | AI & ML Dept | Jai Bharath College")

# --- 3. AUTOMATIC API KEY (STREAMLIT SECRETS) ---
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Missing API Key! Please add GOOGLE_API_KEY to your Streamlit Cloud Secrets.")
    st.stop()

# --- 4. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. CACHED KNOWLEDGE BASE (RAG) ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base(_key):
    folder = "programming_c"
    db_dir = "./chroma_db_c"
    
    if not os.path.exists(folder):
        os.makedirs(folder)
        return None, 0

    loader = PyPDFDirectoryLoader(f"{folder}/")
    docs = loader.load()
    
    if not docs:
        return None, 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # UPDATED: Use the latest stable model and specified task_type
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=_key,
        task_type="retrieval_document"
    )

    if os.path.exists(db_dir):
        try: shutil.rmtree(db_dir)
        except: pass

    # UPDATED: Use the singular 'embedding' keyword
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=db_dir
    )
    return vector_db, len(docs)

# --- 6. INITIALIZE AI ---
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

with st.spinner("LUNA is loading C Programming Knowledge..."):
    vector_db, doc_count = load_knowledge_base(api_key)

# --- 7. SIDEBAR ---
with st.sidebar:
    st.title("🌙 LUNA Settings")
    if vector_db:
        st.success(f"📚 {doc_count} C-Notes Pages Loaded")
    else:
        st.warning("No PDFs found in '/programming_c'.")
    
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("Developed by Abhay Krishna MU")

# --- 8. CHAT INTERFACE ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_query = st.chat_input("Ask a C question (e.g., Explain pointers)")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").markdown(user_query)

    with st.spinner("LUNA is thinking..."):
        system_prompt = """You are LUNA, an expert C tutor for KTU students. 
        Always use C code blocks for code snippets. 
        Explain logic step-by-step. 
        Use LaTeX for mathematical notation ($O(n)$)."""
        
        if vector_db:
            docs = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            full_prompt = f"{system_prompt}\n\nContext from Syllabus Notes:\n{context}\n\nQuestion: {user_query}"
        else:
            full_prompt = f"{system_prompt}\n\nQuestion: {user_query}"
            
        response = model.generate_content(full_prompt)
        answer = response.text

    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

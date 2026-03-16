# --- 1. CHROMADB SQLITE FIX (FOR STREAMLIT CLOUD) ---
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
st.set_page_config(
    page_title="LUNA AI - C Tutor",
    page_icon="💻",
    layout="wide"
)

st.title("💻 LUNA AI: C Programming Tutor")
st.caption("KTU Engineering | AI & ML Department | Jai Bharath College")

# --- 3. LOAD API KEY FROM STREAMLIT SECRETS ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("API key not found. Please add GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- 4. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. LOAD KNOWLEDGE BASE (RAG) ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base(_api_key):
    # This matches your nested GitHub structure: programming_c/programming_c
    folder = "programming_c/programming_c"
    db_dir = "./chroma_db_c"

    if not os.path.exists(folder):
        os.makedirs(folder)
        return None, 0

    # Load PDFs
    loader = PyPDFDirectoryLoader(f"{folder}/")
    docs = loader.load()

    if not docs:
        return None, 0

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # UPDATED EMBEDDINGS LOGIC
    # Using 'text-embedding-004' and specific task types to prevent ValueErrors
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=_api_key,
        task_type="retrieval_document" # Correctly labels vectors for storage
    )

    # Wipe old DB to prevent locked folder errors on 8GB RAM
    if os.path.exists(db_dir):
        try: shutil.rmtree(db_dir)
        except: pass

    # Build the database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings, # Singular 'embedding' is the latest standard
        persist_directory=db_dir
    )

    return vector_db, len(docs)

# --- 6. INITIALIZE KNOWLEDGE BASE ---
with st.spinner("LUNA is indexing your C Programming notes..."):
    vector_db, doc_count = load_knowledge_base(api_key)

# --- 7. SIDEBAR ---
with st.sidebar:
    st.title("🌙 LUNA Settings")
    if vector_db:
        st.success(f"📚 {doc_count} pages of C notes loaded")
    else:
        st.warning("No PDFs found in 'programming_c/programming_c' folder")

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Developed by Abhay Krishna MU")

# --- 8. CHAT DISPLAY & INPUT ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a C programming question...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").markdown(user_query)

    with st.spinner("LUNA is thinking..."):
        system_prompt = """You are LUNA, an expert C tutor for KTU students. 
        Always use C code blocks for code snippets. Explain logic step-by-step. 
        Use LaTeX for mathematical notation like $O(n)$."""
        
        if vector_db:
            # Search context
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
   

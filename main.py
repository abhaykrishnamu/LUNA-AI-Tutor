# --- 1. CORE IMPORTS (Streamlit must be first!) ---
import streamlit as st
import os
import shutil

# --- 2. CHROMADB SQLITE FIX (Required for Streamlit Cloud) ---
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# --- 3. ADDITIONAL LIBRARIES ---
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- 4. PAGE CONFIG ---
st.set_page_config(page_title="LUNA AI - C Tutor", page_icon="🌙", layout="wide")

# --- 5. API SETUP ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("API key not found. Please add GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# FIX: Using the full model path to prevent 404 errors
model = genai.GenerativeModel("models/gemini-1.5-flash")

# --- 6. LOAD KNOWLEDGE BASE ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base(_api_key):
    db_dir = "./chroma_db_c"
    
    # Finds PDF files in the root folder
    pdf_files = [f for f in os.listdir(".") if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        return None, 0

    all_docs = []
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(pdf)
            all_docs.extend(loader.load())
        except Exception:
            continue 

    if not all_docs:
        return None, 0

    # Split text into chunks for RAG
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)

    if not chunks:
        return None, 0

    # Create Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=_api_key,
        task_type="retrieval_document"
    )

    # Clear old database safely
    if os.path.exists(db_dir):
        try:
            shutil.rmtree(db_dir)
        except:
            pass

    # Build Vector Database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )

    return vector_db, len(pdf_files)

# --- 7. UI LAYOUT ---
st.title("🌙 LUNA AI: C Programming Tutor")
st.caption("KTU Engineering | AI & ML Department | Jai Bharath College")

with st.spinner("LUNA is preparing your KTU notes..."):
    vector_db, doc_count = load_knowledge_base(api_key)

# --- 8. SIDEBAR ---
with st.sidebar:
    st.header("🌙 LUNA Settings")
    if vector_db:
        st.success(f"📚 {doc_count} KTU Note(s) loaded.")
    else:
        st.warning("No PDFs found. Upload your C notes to the GitHub root folder.")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Developed by Abhay Krishna MU")

# --- 9. CHAT SYSTEM ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a C programming question...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("LUNA is thinking..."):
        # Specialized KTU Prompting
        system_prompt = (
            "You are LUNA, an expert C tutor for KTU students (2024 Scheme). "
            "Explain concepts simply and always provide code snippets in clean C blocks. "
            "If context is provided from notes, prioritize that information for the answer."
        )
        
        if vector_db:
            docs = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            full_prompt = f"{system_prompt}\n\nContext from KTU Notes:\n{context}\n\nQuestion: {user_query}"
        else:
            full_prompt = f"{system_prompt}\n\nQuestion: {user_query}"
            
        try:
            response = model.generate_content(full_prompt)
            if response.text:
                answer = response.text
            else:
                answer = "LUNA is silent... try asking again."
        except Exception as e:
            answer = f"LUNA API Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

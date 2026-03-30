# --- 1. CORE IMPORTS (Streamlit must be first to prevent NameError) ---
import streamlit as st
import os
import shutil
import random

# --- 2. CHROMADB SQLITE FIX (Required for Streamlit Cloud environment) ---
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

# FIX: Using 'models/' prefix to ensure the 404 error is resolved
model = genai.GenerativeModel("models/gemini-1.5-flash")

# --- 6. LOAD KNOWLEDGE BASE (RAG Logic) ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base(_api_key):
    db_dir = "./chroma_db_c"
    
    # Looks for PDF notes in the root folder of your GitHub repo
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

    # Split text into chunks for efficient searching
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)

    if not chunks:
        return None, 0

    # Create Embeddings using Google's latest model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=_api_key,
        task_type="retrieval_document"
    )

    # Clear old database safely to prevent conflicts
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

# --- 9. CHAT SYSTEM WITH PERSONALITY ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    # Professional & Witty Intro messages
    intros = [
        "👋 **Welcome, Scholar!** LUNA here. Ready to conquer C Programming? Remember: Coding is 10% writing and 90% wondering why it doesn't work. Let's get to work!",
        "🌙 **LUNA Activated.** I've got your KTU notes ready. Don't worry about the exam; you've got this. Besides, 'C' is the only grade that's also a language!",
        "🚀 **Hello!** I'm LUNA, your AI Tutor. Whether it's Pointers or Structures, we'll solve it together. Let's make this Semester count!",
        "💻 **Systems Check: All Green.** Hello Abhay Krishna MU! Ready to debug your life? Just kidding—let's start with debugging some C code first."
    ]
    initial_msg = random.choice(intros)
    st.session_state.messages.append({"role": "assistant", "content": initial_msg})

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
        # The Persona Prompt
        system_prompt = (
            "You are LUNA, a professional, motivating, and slightly witty C tutor for KTU students (2024 Scheme). "
            "1. Be encouraging and use a 'mentor' tone. "
            "2. Use tech-related humor or puns where appropriate. "
            "3. Provide clear C code blocks for all examples. "
            "4. If the student seems stressed, give a quick motivational tip about their B.Tech journey. "
            "5. Prioritize context from the uploaded KTU notes for syllabus accuracy."
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
                answer = "LUNA is currently processing some background tasks. Can you ask that again?"
        except Exception as e:
            answer = f"LUNA hit a snag: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

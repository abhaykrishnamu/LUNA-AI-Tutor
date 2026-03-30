# --- 1. CORE IMPORTS ---
import streamlit as st
import os
import shutil
import random
import time
import sys
import gc

# --- 2. CHROMADB SQLITE FIX ---
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# --- 3. LIBRARIES ---
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- 4. PAGE CONFIG ---
st.set_page_config(
    page_title="LUNA AI - C Tutor",
    page_icon="🌙",
    layout="wide"
)

# --- 5. API SETUP ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Add GOOGLE_API_KEY in Streamlit Secrets")
    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# ✅ WORKING MODEL (NO MORE 404 ERROR)
model = genai.GenerativeModel("gemini-pro")

# --- 6. OCR FUNCTION (DISABLED SAFE VERSION) ---
def perform_ocr_on_pdf(pdf_path):
    # Gemini Pro (v1beta) does not support file OCR → disable to avoid crash
    return ""

# --- 7. LOAD KNOWLEDGE BASE ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base(_api_key):
    db_dir = "./chroma_db_c"

    pdf_files = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]

    if not pdf_files:
        return None, 0

    all_text = ""

    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(os.path.join(".", pdf))
            docs = loader.load()

            text = " ".join([d.page_content for d in docs])

            # OCR fallback skipped (disabled)
            if len(text.strip()) < 100:
                st.warning(f"⚠️ {pdf} may be scanned. OCR disabled.")

            all_text += text + "\n\n"

        except Exception as e:
            st.warning(f"Error reading {pdf}: {e}")
            continue

    if not all_text.strip():
        return None, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = splitter.split_text(all_text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=_api_key
    )

    # Clean DB safely
    if os.path.exists(db_dir):
        gc.collect()
        try:
            shutil.rmtree(db_dir)
        except Exception as e:
            st.warning(f"DB cleanup failed: {e}")

    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )

    return vector_db, len(pdf_files)

# --- 8. UI ---
st.title("🌙 LUNA AI: C Programming Tutor")
st.caption("KTU Engineering | AI & ML Department")

with st.spinner("📚 LUNA analyzing notes..."):
    vector_db, doc_count = load_knowledge_base(api_key)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🌙 LUNA Settings")

    if vector_db:
        st.success(f"📚 {doc_count} PDF(s) loaded")
    else:
        st.warning("⚠️ No readable PDFs found")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Developed by Abhay 🚀")

# --- 9. CHAT SYSTEM ---
if "messages" not in st.session_state:
    st.session_state.messages = []

    intros = [
        "👋 Welcome! I can read your notes.",
        "🌙 LUNA ready. Ask me anything about C programming.",
        "🚀 Let's master C together!"
    ]

    st.session_state.messages.append({
        "role": "assistant",
        "content": random.choice(intros)
    })

# Show chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_query = st.chat_input("Ask a C programming question...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("🤖 LUNA thinking..."):

        system_prompt = (
            "You are LUNA, a smart and friendly C programming tutor. "
            "Explain clearly with examples."
        )

        if vector_db:
            docs = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([d.page_content for d in docs])

            prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {user_query}"
        else:
            st.info("⚠️ Answering without notes (PDF empty or scanned).")
            prompt = f"{system_prompt}\n\nQuestion: {user_query}"

        try:
            response = model.generate_content(prompt)
            answer = getattr(response, "text", "⚠️ No response generated.")
        except Exception as e:
            answer = f"❌ Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

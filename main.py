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

# --- 3. ADDITIONAL LIBRARIES ---
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    st.error("❌ API key not found. Add GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# --- 6. OCR FUNCTION (SAFE VERSION) ---
def perform_ocr_on_pdf(pdf_path):
    try:
        if not hasattr(genai, "upload_file"):
            return ""

        sample_file = genai.upload_file(path=pdf_path)

        max_wait = 30
        waited = 0

        while sample_file.state.name == "PROCESSING" and waited < max_wait:
            time.sleep(2)
            waited += 2
            sample_file = genai.get_file(sample_file.name)

        if waited >= max_wait:
            return ""

        response = model.generate_content([
            sample_file,
            "Extract all readable text clearly. Preserve code formatting."
        ])

        genai.delete_file(sample_file.name)

        return getattr(response, "text", "")

    except Exception as e:
        st.warning(f"OCR failed for {pdf_path}: {e}")
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

            text_content = " ".join([d.page_content for d in docs])

            if len(text_content.strip()) < 100:
                with st.spinner(f"🔍 OCR reading {pdf}..."):
                    text_content = perform_ocr_on_pdf(pdf)

            all_text += text_content + "\n\n"

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

    # Clean old DB safely
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
st.caption("KTU Engineering | AI & ML Department | Jai Bharath College")

with st.spinner("📚 LUNA is analyzing your notes..."):
    vector_db, doc_count = load_knowledge_base(api_key)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🌙 LUNA Settings")

    if vector_db:
        st.success(f"📚 {doc_count} PDF(s) loaded")
    else:
        st.warning("⚠️ No PDFs found")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Developed by Abhay Krishna MU")

# --- 9. CHAT SYSTEM ---
if "messages" not in st.session_state:
    st.session_state.messages = []

    intros = [
        "👋 Welcome! I can read your notes—even scanned ones.",
        "🌙 LUNA ready. Ask anything about C programming.",
        "🚀 Let's study C with your KTU notes!"
    ]

    st.session_state.messages.append({
        "role": "assistant",
        "content": random.choice(intros)
    })

# Display chat
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

            full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {user_query}"
        else:
            st.info("⚠️ Answering without notes...")
            full_prompt = f"{system_prompt}\n\nQuestion: {user_query}"

        try:
            response = model.generate_content(full_prompt)
            answer = getattr(response, "text", "⚠️ No response generated.")
        except Exception as e:
            answer = f"❌ Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

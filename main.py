# --- IMPORTS ---
import streamlit as st
import os
import shutil
import sys
import time

# SQLITE FIX
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except:
    pass

# LIBRARIES
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# PAGE
st.set_page_config(page_title="🌙 LUNA AI", layout="wide")

# API
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Add GOOGLE_API_KEY")
    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# ✅ FIXED MODEL
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# OCR
def perform_ocr_on_pdf(pdf_path):
    try:
        file = genai.upload_file(path=pdf_path)

        timeout = 30
        start = time.time()

        while file.state.name == "PROCESSING":
            if time.time() - start > timeout:
                return ""
            time.sleep(2)
            file = genai.get_file(file.name)

        response = model.generate_content([file, "Extract all text"])

        return response.text if response else ""
    except Exception as e:
        st.warning(f"OCR error: {e}")
        return ""

# LOAD DATA
@st.cache_resource
def load_knowledge_base(api_key):
    db_dir = "./chroma_db_c"

    if not os.path.exists("notes"):
        return None, 0

    pdf_files = [os.path.join("notes", f) for f in os.listdir("notes") if f.endswith(".pdf")]

    st.write("📂 PDFs:", pdf_files)

    all_text = ""

    for pdf in pdf_files:
        st.write("📄 Processing:", pdf)

        try:
            loader = PyPDFLoader(pdf)
            docs = loader.load()
            text = " ".join([d.page_content for d in docs])

            if len(text.strip()) < 150:
                st.warning("🔍 Using OCR...")
                text = perform_ocr_on_pdf(pdf)

            if not text.strip():
                continue

            all_text += text

        except Exception as e:
            st.error(f"Error: {e}")

    if not all_text.strip():
        return None, 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(all_text)

    if not chunks:
        return None, 0

    # ✅ FIXED EMBEDDINGS
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    vector_db = Chroma.from_texts(chunks, embeddings, persist_directory=db_dir)

    return vector_db, len(pdf_files)

# UI
st.title("🌙 LUNA AI: C Tutor")

if "db_loaded" not in st.session_state:
    with st.spinner("📚 Reading notes..."):
        vector_db, doc_count = load_knowledge_base(api_key)
        st.session_state.vector_db = vector_db
        st.session_state.db_loaded = True
else:
    vector_db = st.session_state.vector_db

# CHAT
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything!"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask a question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    try:
        if vector_db:
            docs = vector_db.similarity_search(query, k=3)
            context = "\n".join([d.page_content for d in docs])
            prompt = f"{context}\n\n{query}"
        else:
            prompt = query

        response = model.generate_content(prompt)
        answer = response.text

    except Exception as e:
        answer = f"❌ Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

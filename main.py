# --- 1. CORE IMPORTS ---
import streamlit as st
import os
import shutil
import sys
import gc

# --- 2. SQLITE FIX FOR CHROMA ---
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except:
    pass

# --- 3. LIBRARIES ---
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- 4. PAGE CONFIG ---
st.set_page_config(page_title="🌙 LUNA AI - C Tutor", layout="wide")

# --- 5. API SETUP ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Please add GOOGLE_API_KEY in Streamlit Secrets")
    st.stop()

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- 6. LOAD MODEL (STABLE) ---
def load_model():
    try:
        return genai.GenerativeModel("gemini-1.5-flash")
    except:
        return None

model = load_model()

# --- 7. LOAD KNOWLEDGE BASE (NO OCR, FAST) ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base(api_key):
    db_dir = "./chroma_db_c"

    # 📂 Load PDFs ONLY from "notes" folder
    notes_path = "notes"
    if not os.path.exists(notes_path):
        return None, 0

    pdf_files = [os.path.join(notes_path, f) for f in os.listdir(notes_path) if f.endswith(".pdf")]

    if not pdf_files:
        return None, 0

    all_text = ""

    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(pdf)
            docs = loader.load()

            text = " ".join([d.page_content for d in docs])

            # 🚫 Skip scanned PDFs (no OCR)
            if len(text.strip()) < 150:
                st.warning(f"⚠️ Skipping scanned PDF: {pdf}")
                continue

            all_text += text + "\n\n"

        except Exception as e:
            st.warning(f"Error reading {pdf}")
            continue

    if not all_text.strip():
        return None, 0

    # ✂️ Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_text(all_text)

    # 🧠 Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

    # 🧹 Clean old DB
    if os.path.exists(db_dir):
        try:
            shutil.rmtree(db_dir)
        except:
            pass

    # 💾 Create DB
    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )

    return vector_db, len(pdf_files)

# --- 8. UI ---
st.title("🌙 LUNA AI: C Programming Tutor")
st.caption("KTU Engineering | AI & ML Department")

# --- LOAD ONLY ONCE ---
if "db_loaded" not in st.session_state:
    with st.spinner("📚 LUNA is reading your notes..."):
        vector_db, doc_count = load_knowledge_base(st.secrets["GOOGLE_API_KEY"])
        st.session_state.vector_db = vector_db
        st.session_state.doc_count = doc_count
        st.session_state.db_loaded = True
else:
    vector_db = st.session_state.vector_db
    doc_count = st.session_state.doc_count

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

# --- CHAT MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 I've analyzed your notes! Ask me anything from your syllabus."}
    ]

# --- DISPLAY CHAT ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- USER INPUT ---
user_query = st.chat_input("Ask a C programming question...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("🤖 LUNA thinking..."):

        system_prompt = "You are LUNA, a friendly C programming tutor. Answer clearly based on student's notes."

        if vector_db:
            docs = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([d.page_content for d in docs])

            prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {user_query}"
        else:
            prompt = f"{system_prompt}\n\nQuestion: {user_query}"

        try:
            response = model.generate_content(prompt)
            answer = response.text if response else "⚠️ No response generated."
        except Exception as e:
            answer = "❌ Error generating response. Please try again."

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# --- 1. IMPORTS ---
import streamlit as st
import os
import shutil
import sys
import time

# --- 2. SQLITE FIX ---
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except:
    pass

# --- 3. LIBRARIES ---
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings   # ✅ FIXED
from langchain_community.vectorstores import Chroma

# --- 4. PAGE CONFIG ---
st.set_page_config(page_title="🌙 LUNA AI Tutor", layout="wide")

# --- 5. API KEY ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Add GOOGLE_API_KEY in Streamlit Secrets")
    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# --- 6. AUTO MODEL DETECTION ---
def load_model():
    try:
        models = genai.list_models()
        for m in models:
            if "generateContent" in m.supported_generation_methods:
                return genai.GenerativeModel(m.name)
        return None
    except:
        return None

model = load_model()

# --- 7. LUNA SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are LUNA 🌙, a friendly and intelligent C programming tutor.

Your job:
- Explain concepts step-by-step
- Use simple language
- Give examples

Style:
- Friendly teacher
- Use bullet points
- Help understanding
"""

# --- 8. OCR FUNCTION ---
def perform_ocr_on_pdf(pdf_path):
    try:
        if not model:
            return ""

        file = genai.upload_file(path=pdf_path)

        timeout = 30
        start = time.time()

        while file.state.name == "PROCESSING":
            if time.time() - start > timeout:
                return ""
            time.sleep(2)
            file = genai.get_file(file.name)

        response = model.generate_content([file, "Extract all readable text"])

        genai.delete_file(file.name)

        return response.text if response else ""

    except:
        return ""

# --- 9. LOAD KNOWLEDGE BASE ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    db_dir = "./chroma_db_c"

    if not os.path.exists("notes"):
        return None, 0

    pdf_files = [os.path.join("notes", f) for f in os.listdir("notes") if f.endswith(".pdf")]

    if not pdf_files:
        return None, 0

    all_text = ""

    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(pdf)
            docs = loader.load()
            text = " ".join([d.page_content for d in docs])

            # OCR if scanned
            if len(text.strip()) < 150:
                text = perform_ocr_on_pdf(pdf)

            # fallback
            if not text.strip():
                text = "C programming basics include variables, loops, arrays, functions."

            all_text += text + "\n\n"

        except:
            continue

    if not all_text.strip():
        return None, 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(all_text)

    if not chunks:
        return None, 0

    # ✅ LOCAL EMBEDDINGS (NO ERROR)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)

    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )

    return vector_db, len(pdf_files)

# --- 10. UI ---
st.title("🌙 LUNA AI: C Programming Tutor")

# SAFE LOAD
if "db_loaded" not in st.session_state:
    with st.spinner("📚 Reading notes..."):
        vector_db, doc_count = load_knowledge_base()

        st.session_state.vector_db = vector_db if vector_db else None
        st.session_state.doc_count = doc_count if doc_count else 0
        st.session_state.db_loaded = True

vector_db = st.session_state.get("vector_db", None)
doc_count = st.session_state.get("doc_count", 0)

# SIDEBAR
with st.sidebar:
    st.header("🌙 LUNA")

    if vector_db:
        st.success(f"📚 {doc_count} PDFs loaded")
    else:
        st.warning("⚠️ No notes loaded")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# CHAT MEMORY
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi, I'm LUNA 🌙 — your C programming tutor!"}
    ]

# DISPLAY CHAT
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# USER INPUT
query = st.chat_input("Ask a C programming question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("🤖 LUNA thinking..."):

        try:
            if vector_db:
                docs = vector_db.similarity_search(query, k=3)
                context = "\n\n".join([d.page_content for d in docs])

                prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Answer:
"""
            else:
                prompt = f"""
{SYSTEM_PROMPT}

Question:
{query}

Answer:
"""

            if model:
                response = model.generate_content(prompt)
                answer = response.text if response else "⚠️ No response"
            else:
                answer = "❌ Model not available"

        except Exception as e:
            answer = f"❌ Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

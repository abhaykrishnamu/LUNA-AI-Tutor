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
    import sys
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
st.set_page_config(page_title="LUNA AI - C Tutor", page_icon="🌙", layout="wide")

# --- 5. API SETUP ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Add GOOGLE_API_KEY in Streamlit Secrets")
    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# --- 6. AUTO MODEL DETECTION ---
def get_working_model():
    try:
        models = genai.list_models()
        for m in models:
            if "generateContent" in m.supported_generation_methods:
                return genai.GenerativeModel(m.name)
        return None
    except Exception as e:
        return None

model = get_working_model()

# --- 7. FIXED OCR FUNCTION ---
def perform_ocr_on_pdf(pdf_path):
    """Uses Gemini Vision to read scanned PDF pages."""
    try:
        # Upload to Google File API
        sample_file = genai.upload_file(path=pdf_path)
        
        # Wait for processing
        while sample_file.state.name == "PROCESSING":
            time.sleep(2)
            sample_file = genai.get_file(sample_file.name)

        # Use the model to 'read' the images
        response = model.generate_content([sample_file, "Extract all text and code from these notes."])
        genai.delete_file(sample_file.name)
        return response.text
    except Exception:
        return ""

# --- 8. LOAD KNOWLEDGE BASE ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base(_api_key):
    db_dir = "./chroma_db_c"
    pdf_files = [f for f in os.listdir(".") if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        return None, 0

    all_text = ""
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(pdf)
            docs = loader.load()
            text = " ".join([d.page_content for d in docs])

            # Trigger OCR if page text is too short (indicates a scan)
            if len(text.strip()) < 150:
                text = perform_ocr_on_pdf(pdf)

            all_text += text + "\n\n"
        except:
            continue

    if not all_text.strip():
        return None, 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_text(all_text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=_api_key
    )

    if os.path.exists(db_dir):
        gc.collect()
        try: shutil.rmtree(db_dir)
        except: pass

    vector_db = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=db_dir)
    return vector_db, len(pdf_files)

# --- 9. UI & CHAT ---
st.title("🌙 LUNA AI: C Programming Tutor")
st.caption("KTU Engineering | AI & ML Department")

with st.spinner("📚 LUNA is reading your notes (this may take a minute for scans)..."):
    vector_db, doc_count = load_knowledge_base(api_key)

with st.sidebar:
    st.header("🌙 LUNA Settings")
    if vector_db:
        st.success(f"📚 {doc_count} PDF(s) loaded")
    else:
        st.warning("⚠️ No readable PDFs found")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 I've analyzed your KTU notes! Ask me anything."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a C programming question...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("🤖 LUNA thinking..."):
        system_prompt = "You are LUNA, a friendly C tutor. Use the following context from the student's notes."
        if vector_db:
            docs = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {user_query}"
        else:
            prompt = f"{system_prompt}\n\nQuestion: {user_query}"

        try:
            response = model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            answer = f"❌ Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

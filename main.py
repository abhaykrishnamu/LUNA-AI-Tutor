# --- 1. CORE IMPORTS ---
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
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- 4. PAGE CONFIG ---
st.set_page_config(page_title="🌙 LUNA AI - C Tutor", layout="wide")

# --- 5. API SETUP ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Add GOOGLE_API_KEY in Streamlit Secrets")
    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# --- 6. LOAD MODEL ---
def load_model():
    try:
        return genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.5,
                "max_output_tokens": 2048,
            }
        )
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

model = load_model()

# --- 7. OCR FUNCTION (SAFE) ---
def perform_ocr_on_pdf(pdf_path):
    try:
        if not model:
            return ""

        file = genai.upload_file(path=pdf_path)

        timeout = 40
        start = time.time()

        while file.state.name == "PROCESSING":
            if time.time() - start > timeout:
                return ""
            time.sleep(2)
            file = genai.get_file(file.name)

        response = model.generate_content([
            file,
            "Extract all readable text and code from this document."
        ])

        genai.delete_file(file.name)

        return response.text if response else ""

    except Exception as e:
        st.warning(f"OCR failed: {e}")
        return ""

# --- 8. LOAD KNOWLEDGE BASE ---
@st.cache_resource(show_spinner=False)
def load_knowledge_base(api_key):
    db_dir = "./chroma_db_c"
    notes_path = "notes"

    if not os.path.exists(notes_path):
        st.warning("⚠️ 'notes' folder not found")
        return None, 0

    pdf_files = [os.path.join(notes_path, f) for f in os.listdir(notes_path) if f.endswith(".pdf")]

    st.write("📂 PDFs found:", pdf_files)

    if not pdf_files:
        return None, 0

    all_text = ""

    for pdf in pdf_files:
        st.write(f"📄 Processing: {pdf}")

        try:
            loader = PyPDFLoader(pdf)
            docs = loader.load()
            text = " ".join([d.page_content for d in docs])

            # If scanned → OCR
            if len(text.strip()) < 150:
                st.warning(f"🔍 Using OCR for: {pdf}")
                text = perform_ocr_on_pdf(pdf)

                st.write(f"OCR text length: {len(text)}")

                # Fallback if OCR fails
                if not text.strip():
                    text = "C programming basics include variables, loops, functions, arrays, and pointers."

            all_text += text + "\n\n"

        except Exception as e:
            st.error(f"Error reading {pdf}: {e}")
            continue

    if not all_text.strip():
        return None, 0

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_text(all_text)

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

    # Clear DB
    if os.path.exists(db_dir):
        try:
            shutil.rmtree(db_dir)
        except:
            pass

    vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )

    return vector_db, len(pdf_files)

# --- 9. UI ---
st.title("🌙 LUNA AI: C Programming Tutor")
st.caption("KTU Engineering | AI & ML Department")

# Load once
if "db_loaded" not in st.session_state:
    with st.spinner("📚 LUNA is reading your notes..."):
        vector_db, doc_count = load_knowledge_base(api_key)
        st.session_state.vector_db = vector_db
        st.session_state.doc_count = doc_count
        st.session_state.db_loaded = True
else:
    vector_db = st.session_state.vector_db
    doc_count = st.session_state.doc_count

# Sidebar
with st.sidebar:
    st.header("🌙 LUNA Settings")

    if vector_db is not None:
        st.success(f"📚 {doc_count} PDF(s) loaded")
    else:
        st.warning("⚠️ No readable PDFs found")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 I've analyzed your notes! Ask me anything."}
    ]

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

        if vector_db is not None:
            docs = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            prompt = f"You are a C programming tutor.\n\nContext:\n{context}\n\nQuestion: {user_query}"
        else:
            prompt = f"You are a helpful C programming tutor.\n\nQuestion: {user_query}"

        try:
            response = model.generate_content(prompt)

            if response and hasattr(response, "text"):
                answer = response.text
            else:
                answer = "⚠️ Empty response from AI."

        except Exception as e:
            answer = f"❌ Gemini Error: {str(e)}"
            st.error(answer)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

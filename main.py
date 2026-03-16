import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

# -------------------------------
# 1. PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="LUNA Chemistry Tutor",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 LUNA AI Chemistry Tutor")
st.caption("KTU Engineering Chemistry | 2024 Scheme Study Assistant")

# -------------------------------
# 2. SESSION STATE
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quiz_mode" not in st.session_state:
    st.session_state.quiz_mode = False

MAX_MESSAGES = 15

# -------------------------------
# 3. SIDEBAR
# -------------------------------
with st.sidebar:
    st.title("🧪 LUNA Controls")
    
    # Use st.secrets for safety if deployed, otherwise text input
    api_key = st.text_input("Gemini API Key", type="password")
    
    mode = st.selectbox(
        "Learning Mode",
        ["Beginner", "Exam Preparation", "Quick Answer"]
    )

    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 Quiz"):
            st.session_state.quiz_mode = True
    with col2:
        if st.button("🧹 Clear"):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("Developed by Abhay Krishna MU | AI & ML")

# -------------------------------
# 4. LOAD VECTOR DATABASE (CACHED)
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_vector_db(_api_key):
    folder = "chemistry"
    db_dir = "./chroma_db_storage"

    # Fix: Ensure folder exists to avoid PyPDFDirectoryLoader crashing
    if not os.path.exists(folder):
        os.makedirs(folder)
        return None, 0

    # Fix: Use explicit path with trailing slash for Linux compatibility
    loader = PyPDFDirectoryLoader(f"{folder}/")
    docs = loader.load()

    if not docs:
        return None, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=_api_key
    )

    # Fix: Ensure old DB is cleared if corrupted
    if os.path.exists(db_dir):
        try:
            shutil.rmtree(db_dir)
        except:
            pass

    vector_db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=db_dir
    )

    return vector_db, len(docs)

# -------------------------------
# 5. AI ENGINE SETUP
# -------------------------------
system_prompt = f"""
You are LUNA, an expert Chemistry tutor for the KTU Engineering syllabus.
Mode: {mode}
Rules:
- Explain concepts step-by-step.
- Use LaTeX for ALL chemical formulas (e.g., $H_2SO_4$, $CH_4$).
- Focus strictly on KTU Engineering Chemistry topics.
- For 'Beginner' mode, use simple analogies.
"""

vector_db = None

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    with st.spinner("Indexing Chemistry Knowledge Base..."):
        vector_db, doc_count = load_vector_db(api_key)
    
    if vector_db:
        st.sidebar.success(f"📚 {doc_count} Pages Indexed")
    else:
        st.sidebar.warning("No PDFs found in '/chemistry' folder.")
else:
    st.warning("Please enter your Gemini API Key in the sidebar to begin.")

# -------------------------------
# 6. CHAT INTERFACE
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask LUNA about Chemistry...")

if user_query and api_key:
    # Display user message
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Limit Memory
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

    with st.spinner("LUNA is analyzing..."):
        # 1. QUIZ MODE
        if st.session_state.quiz_mode:
            prompt = f"{system_prompt}\nGenerate 3-5 exam-style questions with answers for {mode} level."
            response = model.generate_content(prompt)
            answer = response.text
            st.session_state.quiz_mode = False

        # 2. RAG MODE (If PDFs exist)
        elif vector_db:
            search_results = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([d.page_content for d in search_results])
            full_prompt = f"{system_prompt}\nContext:\n{context}\n\nQuestion: {user_query}"
            response = model.generate_content(full_prompt)
            answer = response.text

        # 3. GENERAL AI MODE
        else:
            response = model.generate_content(system_prompt + "\nQuestion: " + user_query)
            answer = response.text

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# -------------------------------
# 7. DOWNLOAD UTILITY
# -------------------------------
if st.session_state.messages:
    chat_log = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
    st.sidebar.download_button("📥 Download History", chat_log, file_name="chemistry_session.txt")

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
    page_title="LUNA C Programming Tutor",
    page_icon="💻",
    layout="wide"
)

st.title("💻 LUNA AI: C Programming Tutor")
st.caption("KTU Engineering - Programming in C (EST 120) | AI Tutor")

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
    st.title("💻 C-Tutor Controls")
    
    api_key = st.text_input("Gemini API Key", type="password")
    
    mode = st.selectbox(
        "Teaching Style",
        ["Beginner (Step-by-Step)", "Debug Mode (Fix my Code)", "Exam Prep"]
    )

    st.divider()
    
    if st.button("🎯 Generate C Quiz"):
        st.session_state.quiz_mode = True

    if st.button("🧹 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Developed by Abhay Krishna MU | AI & ML")

# -------------------------------
# 4. LOAD VECTOR DATABASE (CACHED)
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_vector_db(_api_key):
    folder = "programming_c"  # Changed folder name
    db_dir = "./chroma_db_c"

    if not os.path.exists(folder):
        os.makedirs(folder)
        return None, 0

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
# 5. C-TEACHER SYSTEM PROMPT
# -------------------------------
system_prompt = f"""
You are LUNA, an expert C Programming tutor for KTU Engineering students.
Teaching Style: {mode}

Rules:
- When writing code, always use triple backticks with 'c' for syntax highlighting.
- Explain pointers, arrays, and memory management simply.
- Use LaTeX for any mathematical logic or complexity analysis (e.g., $O(n)$).
- If the user provides code with an error, explain WHY it failed before fixing it.
- Focus on the KTU EST 120 syllabus.
"""

# -------------------------------
# 6. AI ENGINE SETUP
# -------------------------------
vector_db = None

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    with st.spinner("Analyzing C Programming Notes..."):
        vector_db, doc_count = load_vector_db(api_key)
    
    if vector_db:
        st.sidebar.success(f"📚 {doc_count} C-Notes Pages Indexed")
    else:
        st.sidebar.warning("No PDFs found in '/programming_c' folder.")
else:
    st.warning("Please enter your Gemini API Key in the sidebar.")

# -------------------------------
# 7. CHAT INTERFACE
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a C question (e.g., 'What are pointers?')")

if user_query and api_key:
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

    with st.spinner("LUNA is coding..."):
        if st.session_state.quiz_mode:
            prompt = f"{system_prompt}\nGenerate 3 C programming challenges (basic, intermediate, advanced) based on the syllabus."
            response = model.generate_content(prompt)
            answer = response.text
            st.session_state.quiz_mode = False

        elif vector_db:
            search_results = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([d.page_content for d in search_results])
            full_prompt = f"{system_prompt}\nContext from Notes:\n{context}\n\nQuestion: {user_query}"
            response = model.generate_content(full_prompt)
            answer = response.text

        else:
            response = model.generate_content(system_prompt + "\nQuestion: " + user_query)
            answer = response.text

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# -------------------------------
# 8. DOWNLOAD UTILITY
# -------------------------------
if st.session_state.messages:
    chat_log = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
    st.sidebar.download_button("📥 Download Code/Chat", chat_log, file_name="c_study_session.txt")

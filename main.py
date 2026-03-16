# --- 1. CHROMADB SQLITE FIX (FOR STREAMLIT CLOUD) ---
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass


import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os


# -------------------------------
# 2 PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="LUNA AI - C Tutor",
    page_icon="💻",
    layout="wide"
)

st.title("💻 LUNA AI: C Programming Tutor")
st.caption("KTU Engineering | AI & ML Department | Jai Bharath College")


# -------------------------------
# 3 LOAD API KEY FROM STREAMLIT SECRETS
# -------------------------------
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("API key not found. Please add GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")


# -------------------------------
# 4 SESSION STATE
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------
# 5 LOAD KNOWLEDGE BASE (RAG)
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_knowledge_base(_api_key):


folder = "programming_c/programming_c"
    db_dir = "chroma_db_c"

    if not os.path.exists(folder):
        os.makedirs(folder)
        return None, 0

    loader = PyPDFDirectoryLoader(folder)
    docs = loader.load()

    if len(docs) == 0:
        return None, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    if len(chunks) == 0:
        return None, 0

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=_api_key,
        task_type="retrieval_document"
    )

    # Load existing DB if available
    if os.path.exists(db_dir):

        vector_db = Chroma(
            persist_directory=db_dir,
            embedding_function=embeddings
        )

    else:

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_dir
        )

    return vector_db, len(docs)


# -------------------------------
# 6 INITIALIZE KNOWLEDGE BASE
# -------------------------------
with st.spinner("Loading C Programming Knowledge Base..."):
    vector_db, doc_count = load_knowledge_base(api_key)


# -------------------------------
# 7 SIDEBAR
# -------------------------------
with st.sidebar:

    st.title("🌙 LUNA Settings")

    if vector_db:
        st.success(f"📚 {doc_count} pages of C notes loaded")
    else:
        st.warning("No PDFs found in 'programming_c' folder")

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Developed by Abhay Krishna MU")


# -------------------------------
# 8 CHAT HISTORY DISPLAY
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------------
# 9 USER INPUT
# -------------------------------
user_query = st.chat_input("Ask a C programming question...")


if user_query:

    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    st.chat_message("user").markdown(user_query)

    with st.spinner("LUNA is thinking..."):

        system_prompt = """
You are LUNA, an expert C programming tutor for KTU engineering students.

Rules:
- Always provide C code using ```c code blocks
- Explain programs step by step
- Use simple examples
- Explain pointers clearly
- Use LaTeX for complexity like $O(n)$
"""

        if vector_db:

            docs = vector_db.similarity_search(user_query, k=3)

            context = "\n\n".join([d.page_content for d in docs])

            full_prompt = f"""
{system_prompt}

Context from C Programming Notes:
{context}

Question:
{user_query}
"""

        else:

            full_prompt = f"""
{system_prompt}

Question:
{user_query}
"""

        response = model.generate_content(full_prompt)

        answer = response.text


    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

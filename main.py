import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from PIL import Image
import os

# -------------------------------
# 1 PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="LUNA - KTU AI Tutor",
    page_icon="🌙",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color:#0e1117;}
.stButton>button {
    width:100%;
    border-radius:10px;
    height:3em;
    background-color:#2e3b4e;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 2 SESSION MEMORY (FIXED)
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quiz_mode" not in st.session_state:
    st.session_state.quiz_mode = False

# -------------------------------
# 3 SIDEBAR
# -------------------------------
with st.sidebar:
    st.title("🌙 LUNA Control")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.divider()
    subject = st.selectbox(
        "Choose Subject",
        ["Chemistry","Maths","Programming in C","IPR","FOC","DM"]
    )

    folder_map = {
        "Chemistry":"chemistry",
        "Maths":"maths",
        "Programming in C":"programming_c",
        "IPR":"ipr",
        "FOC":"foc",
        "DM":"dm"
    }
    current_folder = folder_map[subject]

    st.divider()
    mode = st.selectbox(
        "Learning Mode",
        ["Beginner","Exam Preparation","Quick Answer"]
    )

    if st.button("🎯 Generate Quiz"):
        st.session_state.quiz_mode = True
        st.info("Quiz mode activated! Ask LUNA to 'Start Quiz'.")

# -------------------------------
# 4 SYSTEM PROMPT
# -------------------------------
system_prompt = f"""
You are LUNA, an AI tutor for the KTU 2024 syllabus.
Teaching Mode: {mode}
Rules:
- Explain step-by-step.
- Use simple explanations.
- Use LaTeX for formulas.
- Focus on {subject}.
"""

# -------------------------------
# 5 AI ENGINE (FIXED PERSISTENCE)
# -------------------------------
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    if not os.path.exists(current_folder):
        os.makedirs(current_folder)

    # Fixed: Added "/" to path
    loader = PyPDFDirectoryLoader(f"{current_folder}/")
    docs = loader.load()
    vector_db = None

    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        
        db_path = f"db/{current_folder}"
        
        # Fixed: Simplified Chroma loading
        vector_db = Chroma.from_documents(
            chunks, 
            embeddings, 
            persist_directory=db_path
        )
        st.sidebar.success(f"📚 {len(docs)} pages of {subject} indexed")

# -------------------------------
# 6 MAIN UI
# -------------------------------
st.title(f"🌙 LUNA AI Tutor — {subject}")
st.caption("KTU 2024 Scheme AI Study Assistant | AI & ML Dept")

# Display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

cam_img = st.camera_input("📸 Capture Problem")
user_query = st.chat_input("Ask LUNA something...")

# -------------------------------
# 7 PROCESS USER INPUT
# -------------------------------
if user_query:
    st.session_state.messages.append({"role":"user", "content":user_query})
    st.chat_message("user").markdown(user_query)

    with st.spinner("LUNA is thinking..."):
        # VISION MODE
        if cam_img:
            img = Image.open(cam_img)
            prompt = system_prompt + f"\nSolve this {subject} problem."
            response = model.generate_content([prompt, img])
            answer = response.text

        # QUIZ MODE
        elif st.session_state.quiz_mode:
            prompt = system_prompt + f"\nGenerate 5 exam questions for {subject} based on provided notes. Include answers."
            response = model.generate_content(prompt)
            answer = response.text
            st.session_state.quiz_mode = False # Reset after generation

        # RAG MODE
        elif vector_db:
            search = vector_db.similarity_search(user_query, k=3)
            context = "\n".join([d.page_content for d in search])
            prompt = f"{system_prompt}\nContext:\n{context}\n\nQuestion:\n{user_query}"
            response = model.generate_content(prompt)
            answer = response.text

        # GENERAL AI MODE
        else:
            response = model.generate_content(system_prompt + user_query)
            answer = response.text

    st.session_state.messages.append({"role":"assistant", "content":answer})
    st.chat_message("assistant").markdown(answer)

# -------------------------------
# 8 FOOTER
# -------------------------------
st.divider()
st.caption("Developed by Abhay Krishna MU | AI & ML | Jai Bharath College")
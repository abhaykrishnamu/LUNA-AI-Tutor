# --- IMPORTS ---
import streamlit as st
import google.generativeai as genai

# --- PAGE ---
st.set_page_config(page_title="🌙 LUNA AI", layout="wide")

# --- API KEY ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Please add GOOGLE_API_KEY in Streamlit Secrets")
    st.stop()

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- AUTO MODEL DETECTION ---
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

# --- LUNA SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are LUNA 🌙, a friendly C programming tutor.

- Explain step-by-step
- Use simple language
- Give examples
- Teach like a teacher
"""

# --- UI ---
st.title("🌙 LUNA AI: C Programming Tutor")

# --- CHAT MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi, I'm LUNA 🌙 — your C programming tutor!"}
    ]

# --- SHOW CHAT ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- INPUT ---
query = st.chat_input("Ask a C programming question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("🤖 LUNA thinking..."):

        try:
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

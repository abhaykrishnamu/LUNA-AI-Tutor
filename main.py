# --- 3. API SETUP ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("API key not found. Please add GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# FIX: Added 'models/' prefix which is often required by the newer SDK versions 
# to avoid the 404 error you saw in the screenshot.
model = genai.GenerativeModel("models/gemini-1.5-flash") 

# ... [Keep your Section 4, 5, and 6 the same] ...

# --- 7. CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Ask a C programming question...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("LUNA is thinking..."):
        # Improved System Prompt for KTU Exams
        system_prompt = (
            "You are LUNA, an expert C tutor for KTU students (2024 Scheme). "
            "Explain concepts simply and always provide code snippets in clean C blocks. "
            "If context is provided from notes, prioritize that information."
        )
        
        if vector_db:
            docs = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            full_prompt = f"{system_prompt}\n\nContext from KTU Notes:\n{context}\n\nQuestion: {user_query}"
        else:
            full_prompt = f"{system_prompt}\n\nQuestion: {user_query}"
            
        try:
            # FIX: Adding a safety check for the response
            response = model.generate_content(full_prompt)
            if response.text:
                answer = response.text
            else:
                answer = "LUNA is having trouble generating a response. Please try rephrasing."
        except Exception as e:
            # This will catch and explain any remaining API issues
            answer = f"Oops, LUNA hit a snag: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

import streamlit as st
import requests

# --- CONFIGURATION ---
st.set_page_config(page_title="RAG PDF AI", page_icon="📄", layout="centered")

# IMPORTANT: On Streamlit Cloud, set this in "Secrets"
# Local fallback is for testing
BACKEND_URL = "https://rag-pdf-ai.onrender.com"

st.title("📄 RAG PDF AI")
st.caption(f"Connected to Backend: `{BACKEND_URL}`")

# --- 1. UPLOAD SECTION ---
st.header("1. Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=False)

if uploaded_file is not None:
    if st.button("Start Processing"):
        with st.spinner("Uploading and extracting text..."):
            try:
                # Prepare file for API request
                # We send the file bytes directly to the backend
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                
                # Send to Backend Endpoint
                response = requests.post(f"{BACKEND_URL}/api/upload", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Success! Found {data.get('chunks_count', 0)} chunks.")
                    st.info("The AI is now learning from your document in the background. You can start asking questions in a few seconds.")
                else:
                    st.error(f"Upload failed: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the Backend. Is your Render app running?")
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.divider()

# --- 2. CHAT SECTION ---
st.header("2. Ask a Question")
question = st.text_input("What would you like to know about the document?")
top_k = st.slider("Retrieval Depth (Chunks)", min_value=1, max_value=10, value=5)

if st.button("Ask AI"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            try:
                payload = {"question": question, "top_k": top_k}
                
                # Send question to Backend Endpoint
                response = requests.post(f"{BACKEND_URL}/api/chat", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer provided.")
                    sources = data.get("sources", [])
                    
                    st.markdown("### Answer")
                    st.write(answer)
                    
                    if sources:
                        st.markdown("---")
                        st.caption("Sources:")
                        for source in sources:
                            st.markdown(f"- `{source}`")
                else:
                    st.error(f"Error from API: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the Backend.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

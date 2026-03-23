import streamlit as st
import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

# ✅ API KEY HANDLING (LOCAL + CLOUD)
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    api_key = os.getenv("GROQ_API_KEY")

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ---------------- UI ----------------
st.set_page_config(
    page_title="AI Recruiter Dashboard",
    page_icon="🧠",
    layout="wide",
)

# ---------------- SESSION ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🧠 AI Recruiter")

    uploaded_files = st.file_uploader(
        "Upload resumes (PDF)",
        type="pdf",
        accept_multiple_files=True
    )

    if st.session_state.vectorstore:
        st.success("✅ Resumes processed")
    else:
        st.warning("⏳ Upload resumes")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    if not api_key:
        st.error("❌ GROQ_API_KEY missing")

# ---------------- HEADER ----------------
st.title("📄 AI Resume Screening Chatbot")

# ---------------- PROCESS PDFs (FIXED) ----------------
if uploaded_files:

    # ✅ Detect if new files uploaded
    current_files = sorted([f.name for f in uploaded_files])

    if current_files != st.session_state.uploaded_names:

        # 🔥 RESET OLD DATA
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.uploaded_names = current_files

        with st.spinner("Processing resumes..."):
            documents = []

            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                pages = loader.load()

                for page in pages:
                    page.metadata["source"] = uploaded_file.name

                documents.extend(pages)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_documents(docs, embeddings)
            st.session_state.vectorstore = vectorstore

        st.success(f"✅ {len(uploaded_files)} resume(s) processed!")
        st.rerun()

# ---------------- CHAT ----------------
query = st.chat_input("Ask about candidates...")

# Show chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- QUERY ----------------
if query:
    if not st.session_state.vectorstore:
        st.warning("⚠️ Upload resumes first")
    else:
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching resumes..."):

                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                )

                docs = retriever.invoke(query)

                if not docs:
                    st.warning("⚠️ No relevant data found")
                else:
                    context = "\n\n".join([
                        f"Source: {doc.metadata.get('source')}\n{doc.page_content}"
                        for doc in docs
                    ])

                    llm = ChatGroq(
                        api_key=api_key,
                        model_name="llama-3.1-8b-instant",
                        temperature=0
                    )

                    prompt = f"""
You are an AI recruiter.

From the resumes below, identify best matching candidates.

Return clearly:
- Candidate Name
- Skills
- Experience
- Match Score (%)
- Source File Name

Resumes:
{context}

Question:
{query}
"""

                    response = llm.invoke(prompt)
                    output = response.content

                    st.markdown(output)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": output
                    })
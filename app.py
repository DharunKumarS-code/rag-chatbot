import streamlit as st
import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ─────────────────────────── Page Config ───────────────────────────
st.set_page_config(
    page_title="AI Recruiter Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ────────────────────────────
st.markdown("""
<style>
    /* Global */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    .main { background-color: #f5f7fa; }

    /* Header */
    .dashboard-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .dashboard-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .dashboard-header p  { margin: 0.4rem 0 0; opacity: 0.75; font-size: 1rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        color: white;
    }
    section[data-testid="stSidebar"] * { color: white !important; }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 8px;
    }
    .status-ready   { background: #22c55e; color: #fff !important; }
    .status-waiting { background: #f59e0b; color: #fff !important; }

    /* Chat bubbles */
    .chat-container { max-width: 860px; margin: auto; }

    /* Result cards */
    .result-card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 5px solid #0f3460;
    }
    .result-card h4 { margin: 0 0 0.5rem; color: #1a1a2e; font-size: 1.05rem; }
    .result-card p  { margin: 0.2rem 0; font-size: 0.9rem; color: #444; }
    .score-high   { color: #16a34a; font-weight: 700; }
    .score-medium { color: #d97706; font-weight: 700; }
    .score-low    { color: #dc2626; font-weight: 700; }

    /* Clear button */
    div[data-testid="stButton"] button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Session State ─────────────────────────
if "vectorstore"  not in st.session_state: st.session_state.vectorstore  = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ─────────────────────────── Sidebar ───────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 AI Recruiter")
    st.markdown("---")

    st.markdown("### 📋 How to use")
    st.markdown("""
- Upload one or more **PDF resumes**
- Ask natural language queries, e.g.:
  - *Python developers with 3+ years experience*
  - *Who has AWS and Docker skills?*
  - *Find candidates with ML background*
""")
    st.markdown("---")

    st.markdown("### 📂 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Status badge
    if st.session_state.vectorstore:
        st.markdown('<span class="status-badge status-ready">✅ Resumes processed</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-waiting">⏳ No resumes loaded</span>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # API key warning in sidebar
    if not os.getenv("GROQ_API_KEY"):
        st.error("❌ GROQ_API_KEY missing in .env")

# ─────────────────────────── Header ────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <h1>🧠 AI Recruiter Dashboard</h1>
    <p>Powered by Groq · FAISS · LangChain · Streamlit</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── Process PDFs ──────────────────────────
# (Backend logic unchanged)
if uploaded_files and st.session_state.vectorstore is None:
    with st.spinner("⚙️ Processing resumes — please wait..."):
        documents = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            pages  = loader.load()
            for page in pages:
                page.metadata["source"] = uploaded_file.name
            documents.extend(pages)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        embeddings   = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore  = FAISS.from_documents(docs, embeddings)
        st.session_state.vectorstore = vectorstore

    st.success(f"✅ {len(uploaded_files)} resume(s) processed successfully!")
    st.rerun()

# ─────────────────────────── Chat Area ─────────────────────────────
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Render existing chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        if msg["role"] == "assistant":
            st.markdown(msg["content"], unsafe_allow_html=True)
        else:
            st.write(msg["content"])

# Chat input
query = st.chat_input("Ask about candidates (e.g., Python developers with 3+ years experience)...")

if query:
    # Guard: no resumes uploaded
    if not st.session_state.vectorstore:
        with st.chat_message("assistant", avatar="🤖"):
            st.warning("⚠️ Please upload at least one resume PDF from the sidebar before querying.")
    else:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🧑"):
            st.write(query)

        # Retrieve + generate (backend logic unchanged)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Searching resumes and generating response..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                docs      = retriever.invoke(query)

                if not docs:
                    answer_html = "⚠️ No relevant content found in the uploaded resumes. Try rephrasing your query."
                    st.warning(answer_html)
                else:
                    context = "\n\n".join([
                        f"Source: {doc.metadata.get('source')}\n{doc.page_content}"
                        for doc in docs
                    ])

                    llm = ChatGroq(
                        api_key=os.getenv("GROQ_API_KEY"),
                        model_name="llama-3.1-8b-instant",
                        temperature=0,
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
                    response    = llm.invoke(prompt)
                    raw_content = response.content

                    # ── Format response as styled cards ──────────────────
                    # Split on candidate blocks (each starts with "Candidate" or numbered)
                    import re
                    blocks = re.split(r'\n(?=(?:\d+[\.\)]?\s+)?Candidate\s*(?:Name)?[\s:–-])', raw_content, flags=re.IGNORECASE)

                    if len(blocks) <= 1:
                        # Fallback: render as markdown if no card structure detected
                        answer_html = raw_content
                        st.markdown(answer_html)
                    else:
                        cards_html = ""
                        for block in blocks:
                            if not block.strip():
                                continue

                            # Extract fields with flexible regex
                            name_m  = re.search(r'Candidate(?:\s+Name)?[\s:–-]+(.+)', block, re.IGNORECASE)
                            skill_m = re.search(r'Skills?[\s:–-]+(.+?)(?:\n|Experience|Match)', block, re.IGNORECASE | re.DOTALL)
                            exp_m   = re.search(r'Experience[\s:–-]+(.+?)(?:\n|Match|Source)', block, re.IGNORECASE | re.DOTALL)
                            score_m = re.search(r'Match\s*Score[\s:–-]+(\d+)\s*%?', block, re.IGNORECASE)
                            src_m   = re.search(r'Source(?:\s+File(?:\s+Name)?)?[\s:–-]+(.+)', block, re.IGNORECASE)

                            name   = name_m.group(1).strip()  if name_m  else "Unknown Candidate"
                            skills = skill_m.group(1).strip() if skill_m else "—"
                            exp    = exp_m.group(1).strip()   if exp_m   else "—"
                            score  = int(score_m.group(1))    if score_m else None
                            src    = src_m.group(1).strip()   if src_m   else "—"

                            # Score color
                            if score is None:
                                score_html = "<span>—</span>"
                            elif score >= 75:
                                score_html = f'<span class="score-high">🟢 {score}%</span>'
                            elif score >= 50:
                                score_html = f'<span class="score-medium">🟡 {score}%</span>'
                            else:
                                score_html = f'<span class="score-low">🔴 {score}%</span>'

                            cards_html += f"""
<div class="result-card">
    <h4>👤 {name}</h4>
    <p>🛠️ <b>Skills:</b> {skills}</p>
    <p>📅 <b>Experience:</b> {exp}</p>
    <p>🎯 <b>Match Score:</b> {score_html}</p>
    <p>📄 <b>Source:</b> {src}</p>
</div>"""

                        answer_html = cards_html if cards_html else raw_content
                        st.markdown(answer_html, unsafe_allow_html=True)

                    st.session_state.chat_history.append({"role": "assistant", "content": answer_html})

st.markdown('</div>', unsafe_allow_html=True)

"""
app.py — Streamlit UI for  AI GAP-Chatbot Chatbot (Groq + LLaMA 3.3 70B + ChromaDB + SQLite)
"""
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag.loader import load_file
from rag.chunker import chunk_text
from rag.embedder import embed_texts
from rag.vector_store import add_chunks, reset_store, count_chunks
from rag.retriever import retrieve, format_context
from llm.groq_client import chat
from utils.helpers import (
    init_db, save_document, get_documents, delete_document_record,
    save_message, get_history, clear_history, format_sources, new_session_id
)

# ── Init ────────────────────────────────────────────────────────────────────
init_db()

st.set_page_config(
    page_title=" AI GAP-Chatbot — LLaMA 3.3 via Groq",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0f0f14; color: #e2e8f0; }

section[data-testid="stSidebar"] {
    background: #13131a !important;
    border-right: 1px solid #2d2d3d;
}

.chat-header {
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
    padding: 20px 28px;
    border-radius: 14px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.chat-header h1 { color: white; font-size: 1.5rem; font-weight: 700; margin: 0; }
.chat-header p  { color: rgba(255,255,255,0.8); margin: 0; font-size: 0.85rem; }

.metric-card {
    background: #1a1a26;
    border: 1px solid #2d2d3d;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-card .num { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
.metric-card .lbl { font-size: 0.75rem; color: #94a3b8; margin-top: 2px; }

.doc-chip {
    background: #1e1e2e;
    border: 1px solid #3b3b52;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 4px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
}
.doc-chip .name { color: #c4b5fd; font-weight: 500; }
.doc-chip .meta { color: #64748b; }

.source-badge {
    display: inline-block;
    background: #1e1b4b;
    color: #a5b4fc;
    border: 1px solid #3730a3;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    margin: 2px;
}

[data-testid="stChatMessage"] {
    background: #16161f !important;
    border: 1px solid #2d2d3d !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
}

.stTextInput > div > div > input {
    background: #1a1a26 !important;
    border: 1px solid #3b3b52 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button:hover { opacity: 0.9; }

div[data-testid="stFileUploader"] {
    background: #1a1a26;
    border: 2px dashed #3b3b52;
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = new_session_id()
if "messages" not in st.session_state:
    st.session_state.messages = []

session_id = st.session_state.session_id

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖  AI GAP-Chatbot")
    st.markdown("**Powered by LLaMA 3.3 70B + Groq**")
    st.divider()

    # Metrics
    docs = get_documents()
    chunks = count_chunks()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="num">{len(docs)}</div><div class="lbl">Documents</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="num">{chunks}</div><div class="lbl">Chunks</div></div>', unsafe_allow_html=True)

    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drag & drop or browse",
        type=["pdf", "txt", "md", "docx", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uf in uploaded_files:
            # Check if already uploaded
            existing = [d["filename"] for d in get_documents()]
            if uf.name in existing:
                st.warning(f"'{uf.name}' already uploaded.")
                continue

            with st.spinner(f"Processing {uf.name}…"):
                try:
                    suffix = os.path.splitext(uf.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name

                    text = load_file(tmp_path)
                    os.unlink(tmp_path)

                    chunks_list = chunk_text(text)
                    embeddings = embed_texts(chunks_list)

                    doc_id = save_document(uf.name, suffix.lstrip("."), len(chunks_list))
                    add_chunks(chunks_list, embeddings, uf.name, doc_id)

                    st.success(f"✅ {uf.name} — {len(chunks_list)} chunks")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # Document list
    docs = get_documents()
    if docs:
        st.markdown("### 📄 Uploaded Documents")
        for doc in docs:
            col_n, col_x = st.columns([4, 1])
            with col_n:
                st.markdown(
                    f'<div class="doc-chip">'
                    f'<span class="name">📄 {doc["filename"]}</span>'
                    f'<span class="meta">{doc["chunk_count"]} chunks</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col_x:
                if st.button("🗑️", key=f"del_{doc['id']}", help="Remove document"):
                    from rag.vector_store import delete_document
                    delete_document(doc["id"])
                    delete_document_record(doc["id"])
                    st.rerun()

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Clear Chat"):
            clear_history(session_id)
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("🗑️ Reset All"):
            reset_store()
            clear_history(session_id)
            st.session_state.messages = []
            conn = __import__("sqlite3").connect("chat_history.db")
            conn.execute("DELETE FROM documents")
            conn.commit()
            conn.close()
            st.rerun()

    st.divider()
    st.markdown(
        '<div style="font-size:0.72rem;color:#475569;text-align:center;">'
        'Built with 🤖LaMA 3.3 70B · Groq · ChromaDB · SQLite · Streamlit'
        '</div>',
        unsafe_allow_html=True
    )

# ── Main Chat Area ────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
  <div style="font-size:2rem;"></div>
  <div>
    <h1>🤖AI GAP-Chatbot</h1>
    <p>Ask questions about your documents · Powered by LLaMA 3.3 70B via Groq API</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Load history from DB on first load
if not st.session_state.messages:
    st.session_state.messages = get_history(session_id)

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑"):
        st.markdown(msg["content"])

# Welcome message
if not st.session_state.messages:

    with st.chat_message("assistant", avatar="😎"):
        st.markdown(
            "👋 **Welcome!** I'm your AI GAP-Chatbot powered by **LLaMA 3.3 70B via Groq**.\n\n"
            "📂 Upload documents in the sidebar (PDF, DOCX, TXT, CSV) and ask me anything about them.\n\n"
            "⚡ Groq gives **ultra-fast** responses — try it!"
        )

# ── Chat Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents…"):
    # Show user message
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(session_id, "user", prompt)

    # Retrieve context
    hits = retrieve(prompt) if count_chunks() > 0 else []
    context = format_context(hits)
    sources_text = format_sources(hits)

    # Stream response
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        full_response = ""

        history_for_llm = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        for chunk in chat(history_for_llm, context=context, stream=True):
            full_response += chunk
            placeholder.markdown(full_response + "▌")

        # Add source badges
        if sources_text:
            full_response += f"\n\n{sources_text}"

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_message(session_id, "assistant", full_response)

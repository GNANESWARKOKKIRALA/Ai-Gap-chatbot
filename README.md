# 🤖 AI GAP Chatbot — LLaMA 3.3 70B via Groq

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot built with:
- 🦙 **LLaMA 3.3 70B** — powerful open-source LLM
- ⚡ **Groq API** — world's fastest LLM inference (~300 tokens/sec)
- 🗃️ **ChromaDB** — local vector database for semantic search
- 🗄️ **SQLite** — chat history & document metadata storage
- 📄 **Multi-format support** — PDF, DOCX, TXT, MD, CSV
- 🎨 **Streamlit UI** — clean, dark-themed web interface

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd  AI GAP-Chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get your FREE Groq API Key
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card needed)
3. Go to **API Keys** → Create new key

### 4. Set up environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 5. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser 🎉

---

## 🏗️ Architecture

```
User uploads PDF/DOCX/TXT/CSV
         ↓
   Parse text (loader.py)
         ↓
   Chunk into 500-word pieces (chunker.py)
         ↓
   Embed with sentence-transformers (embedder.py)
         ↓
   Store in ChromaDB (vector_store.py)
         ↓
   User asks a question
         ↓
   Embed query → ChromaDB top-K search (retriever.py)
         ↓
   Inject context into LLaMA 3.3 70B prompt
         ↓
   Stream response via Groq API ⚡ (groq_client.py)
         ↓
   Save to SQLite chat history (helpers.py)
```

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py                  # Streamlit UI entry point
├── .env.example            # Environment template
├── requirements.txt
├── README.md
│
├── rag/
│   ├── loader.py           # PDF, TXT, DOCX, CSV parser
│   ├── chunker.py          # Text chunking with overlap
│   ├── embedder.py         # sentence-transformers embeddings
│   ├── vector_store.py     # ChromaDB CRUD operations
│   └── retriever.py        # Top-K semantic retrieval
│
├── llm/
│   └── groq_client.py      # Groq API + LLaMA 3.3 70B
│
├── utils/
│   └── helpers.py          # SQLite DB, sessions, formatting
│
└── data/
    └── uploads/            # Temporary file storage
```

---

## ⚙️ Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Your Groq API key (required) |
| `LLAMA_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `TOP_K_CHUNKS` | `4` | Number of chunks to retrieve |
| `CHROMA_DB_PATH` | `./chroma_store` | ChromaDB storage path |

---

## 🧠 Tech Decisions

| Decision | Reason |
|---|---|
| **Groq instead of OpenAI** | Free tier, 10x faster, no credit card |
| **LLaMA 3.3 70B** | Open-source, near GPT-4 quality, free |
| **ChromaDB** | Local vector DB, no external service needed |
| **SQLite** | Zero-config persistence for chat history |
| **sentence-transformers** | Free, runs locally, high quality embeddings |

> 💡 Designed to be easily swappable — replace Groq with OpenAI or Claude by editing `llm/groq_client.py`

---

## 🚀 Deployment (Production Upgrade)

```bash
# Switch SQLite → PostgreSQL: update DB_PATH in utils/helpers.py
# Deploy on Railway (free tier):
railway init
railway up
```

---

## 📝 License
MIT — free to use and modify.

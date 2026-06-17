# 🎯 Career Copilot

A **Retrieval-Augmented Generation (RAG)** system that matches hi-tech job listings to a candidate's profile, ranks the fit, and automatically tailors a resume to each role — powered by semantic search and Claude.

Built as a hands-on learning project to master real Python, RAG pipelines, vector databases, and embeddings.

---

## 🧠 What it does

```
Jobs ──embed──▶ Vector DB (Chroma)
                     │
Profile ──embed──▶ Semantic retrieval ──▶ Top matching jobs ──▶ Claude ──▶ Fit score + tailored resume
```

1. **Ingest & embed** job listings into a local vector database
2. **Semantic search** — find the most relevant jobs for a profile by *meaning*, not keywords
3. **Rank & tailor** — Claude scores each match (0–100), explains strengths & gaps, and rewrites the resume summary and bullet points for each specific role

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| RAG framework | LangChain |
| Vector database | Chroma (local, persistent) |
| Embeddings | FastEmbed (`bge-small-en-v1.5`) — runs locally, no API needed |
| Generation | Anthropic Claude (`claude-opus-4-8`) with Structured Outputs |
| PDF parsing | pypdf |

## 📂 Project structure

```
Career-Copilot/
├── jobs.json          # Sample hi-tech job listings
├── stage1_embed.py    # Stage 1: load jobs → embed → store in Chroma
├── stage2_search.py   # Stage 2: semantic search over the jobs
├── stage3_match.py    # Stage 3: rank fit + tailor resume with Claude
├── requirements.txt   # Python dependencies
└── README.md
```

## 🚀 Setup

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

### API key (Stage 3 only)

Stages 1–2 run fully locally with no API key. Stage 3 uses Claude — set your key:

```bash
# Option A: environment variable
set ANTHROPIC_API_KEY=sk-ant-...   # Windows
export ANTHROPIC_API_KEY=sk-ant-... # macOS / Linux

# Option B: create an api_key.txt file in the project root with just the key
```

## ▶️ Usage

```bash
# Stage 1 — build the vector index
python stage1_embed.py

# Stage 2 — semantic search
python stage2_search.py "AI automation roles with an operations background"

# Stage 3 — rank fit + tailor resume (requires API key + a profile.txt)
python stage3_match.py
```

## 💡 Key concepts demonstrated

- **Embeddings** — turning text into vectors that capture meaning
- **Vector similarity search** — retrieving by semantic closeness, not exact words
- **RAG pipeline** — retrieval → ranking → generation
- **Structured Outputs** — forcing the LLM to return a reliable, schema-validated JSON

## 📈 Possible next steps

- Scrape live job listings instead of a static file
- Swap to a multilingual embedding model for better Hebrew support
- Add a web UI (Streamlit / FastAPI)
- Persist match history and track applications

---

Built with curiosity by **Barak Keren** · [GitHub](https://github.com/bke1302)

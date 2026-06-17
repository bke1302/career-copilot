"""
Career Copilot — דמו ציבורי (חיפוש סמנטי בלבד)
================================================
גרסה בטוחה לפריסה ב-Streamlit Cloud:
• רק שלבים 1-2 (הטמעה + חיפוש סמנטי) — רצים מקומית, בחינם, בלי מפתח API.
• שלב 3 (Claude) לא נכלל כאן כדי לא לבזבז קרדיט על מבקרים.

האינדקס נבנה בזיכרון בזמן ריצה מתוך jobs.json (בענן אין chroma_db שמור).

הרצה:   streamlit run app_public.py
"""

import json
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document

BASE_DIR = Path(__file__).parent
JOBS_FILE = BASE_DIR / "jobs.json"


def job_to_document(job: dict) -> Document:
    content = (
        f"תפקיד: {job['title']}\n"
        f"חברה: {job['company']}\n"
        f"בכירות: {job['seniority']}\n"
        f"תיאור: {job['description']}\n"
        f"דרישות: {job['requirements']}"
    )
    return Document(
        page_content=content,
        metadata={
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "seniority": job["seniority"],
            "description": job["description"],
            "requirements": job["requirements"],
        },
    )


@st.cache_resource(show_spinner="בונה אינדקס סמנטי (פעם אחת)...")
def build_vectorstore() -> Chroma:
    jobs = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    docs = [job_to_document(j) for j in jobs]
    embeddings = FastEmbedEmbeddings()
    # אינדקס בזיכרון — ללא persist_directory
    return Chroma.from_documents(documents=docs, embedding=embeddings, collection_name="jobs")


# --- ממשק ---
st.set_page_config(page_title="Career Copilot — Demo", page_icon="🎯", layout="centered")

st.markdown(
    """
    <style>
    .main .block-container { direction: rtl; text-align: right; }
    h1, h2, h3, p, li, label { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎯 Career Copilot — דמו")
st.caption("חיפוש סמנטי במשרות הייטק לפי משמעות (RAG · embeddings · vector search)")

st.info(
    "🔎 זהו דמו ציבורי של מנגנון החיפוש הסמנטי. "
    "הגרסה המלאה מוסיפה דירוג התאמה והתאמת קורות חיים אוטומטית עם Claude.",
    icon="ℹ️",
)

query = st.text_input(
    "מה אתה מחפש?",
    value="משרות AI ואוטומציה עם רקע בתפעול ושירות",
    placeholder="לדוגמה: data engineering בענן",
)
top_k = st.slider("כמה תוצאות?", min_value=1, max_value=8, value=4)

if st.button("🔍 חפש", type="primary") and query.strip():
    vectorstore = build_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    st.subheader(f"{len(results)} המשרות הרלוונטיות ביותר")
    for rank, (doc, distance) in enumerate(results, start=1):
        similarity = max(0.0, 1 - distance) * 100
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"### {doc.metadata['title']}")
                st.caption(f"🏢 {doc.metadata['company']}  ·  📍 {doc.metadata['location']}  ·  {doc.metadata['seniority']}")
            with c2:
                st.metric("התאמה", f"{similarity:.0f}%")
            st.progress(min(1.0, similarity / 100))
            with st.expander("פרטי המשרה"):
                st.write(f"**תיאור:** {doc.metadata['description']}")
                st.write(f"**דרישות:** {doc.metadata['requirements']}")

st.divider()
st.caption("קוד המקור: github.com/bke1302/career-copilot")

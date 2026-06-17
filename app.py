"""
Career Copilot — ממשק Web (Streamlit)
======================================
עוטף את אותו ה-RAG מהסקריפטים בממשק ויזואלי:
מזינים פרופיל → לוחצים כפתור → רואים משרות מדורגות + קו"ח מותאם.

הרצה:
    streamlit run app.py
"""

import json
import os
from pathlib import Path

import anthropic
import streamlit as st
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"
PROFILE_FILE = BASE_DIR / "profile.txt"
COLLECTION_NAME = "jobs"
MODEL = "claude-opus-4-8"

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "match_score": {"type": "integer", "description": "ציון התאמה 0-100"},
        "verdict": {"type": "string", "description": "שורת סיכום אחת על ההתאמה"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "tailored_summary": {"type": "string"},
        "tailored_bullets": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["match_score", "verdict", "strengths", "gaps", "tailored_summary", "tailored_bullets"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "אתה יועץ קריירה מומחה בהייטק. תפקידך לדרג התאמה בין פרופיל מועמד למשרה, "
    "ולהתאים את קורות החיים למשרה הספציפית. היה כן וענייני לגבי פערים — אל תנפח ציונים. "
    "ענה תמיד בעברית. התבסס רק על המידע בפרופיל; אל תמציא ניסיון שלא קיים."
)


# --- משאבים כבדים נטענים פעם אחת בלבד (cache) ---
@st.cache_resource
def load_vectorstore() -> Chroma:
    embeddings = FastEmbedEmbeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )


@st.cache_resource
def load_client() -> anthropic.Anthropic:
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip().strip("﻿").strip()
    key_file = BASE_DIR / "api_key.txt"
    if not key and key_file.exists():
        key = key_file.read_text(encoding="utf-8-sig").strip().strip("﻿").strip()
    if not key:
        return None
    return anthropic.Anthropic(api_key=key)


def analyze_job(client: anthropic.Anthropic, profile: str, job_content: str) -> dict:
    user_message = (
        f"=== הפרופיל שלי (קורות חיים) ===\n{profile}\n\n"
        f"=== המשרה ===\n{job_content}\n\n"
        "דרג את ההתאמה והתאם את קורות החיים למשרה הזו לפי המבנה הנדרש."
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
        output_config={"format": {"type": "json_schema", "schema": ANALYSIS_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def score_color(score: int) -> str:
    if score >= 80:
        return "🟢"
    if score >= 65:
        return "🟡"
    return "🔴"


# --- ממשק ---
st.set_page_config(page_title="Career Copilot", page_icon="🎯", layout="centered")

# עיצוב RTL בסיסי
st.markdown(
    """
    <style>
    .main .block-container { direction: rtl; text-align: right; }
    h1, h2, h3, p, li, label { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎯 Career Copilot")
st.caption("מערכת RAG שמדרגת התאמת משרות לפרופיל שלך ומתאימה את קורות החיים אוטומטית")

# פרופיל ברירת מחדל מהקובץ אם קיים
default_profile = PROFILE_FILE.read_text(encoding="utf-8") if PROFILE_FILE.exists() else ""

profile = st.text_area("📋 הפרופיל / קורות החיים שלך:", value=default_profile, height=220)
top_k = st.slider("כמה משרות לבדוק?", min_value=1, max_value=8, value=3)

if st.button("🔍 מצא משרות מתאימות", type="primary"):
    if not profile.strip():
        st.error("נא להזין פרופיל קודם.")
        st.stop()

    client = load_client()
    if client is None:
        st.error("חסר מפתח API. צור קובץ api_key.txt עם המפתח, או הגדר משתנה סביבה ANTHROPIC_API_KEY.")
        st.stop()

    vectorstore = load_vectorstore()

    with st.spinner("מחפש משרות רלוונטיות (retrieval)..."):
        results = vectorstore.similarity_search_with_score(profile, k=top_k)

    st.success(f"נמצאו {len(results)} משרות רלוונטיות. Claude מנתח כל אחת...")

    for rank, (doc, _distance) in enumerate(results, start=1):
        title = doc.metadata["title"]
        company = doc.metadata["company"]
        location = doc.metadata.get("location", "")

        with st.spinner(f"מנתח: {title} @ {company}..."):
            analysis = analyze_job(client, profile, doc.page_content)

        score = analysis["match_score"]

        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.subheader(f"{title}")
                st.caption(f"🏢 {company}  ·  📍 {location}")
            with c2:
                st.metric("התאמה", f"{score}%")
            st.progress(score / 100)
            st.write(f"{score_color(score)} **{analysis['verdict']}**")

            with st.expander("✅ חוזקות / ⚠️ פערים"):
                st.markdown("**✅ חוזקות:**")
                for s in analysis["strengths"]:
                    st.markdown(f"- {s}")
                st.markdown("**⚠️ פערים:**")
                for g in analysis["gaps"]:
                    st.markdown(f"- {g}")

            with st.expander("📝 קורות חיים מותאמים למשרה"):
                st.markdown("**תקציר מקצועי מותאם:**")
                st.info(analysis["tailored_summary"])
                st.markdown("**בולטים מותאמים:**")
                for b in analysis["tailored_bullets"]:
                    st.markdown(f"- {b}")

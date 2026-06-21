"""
Career Copilot — דמו ציבורי (חיפוש סמנטי בלבד)
================================================
גרסה בטוחה לפריסה ב-Streamlit Cloud, מעוצבת לפי SUCCESS OS Design System.
• רק שלבים 1-2 (הטמעה + חיפוש סמנטי) — רצים מקומית, בחינם, בלי מפתח API.
• האינדקס נבנה בזיכרון בזמן ריצה מתוך jobs.json.
"""

import html
import json
import urllib.parse
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document

BASE_DIR = Path(__file__).parent
JOBS_FILE = BASE_DIR / "jobs.json"

# --- טוקנים מ-SUCCESS OS Design System ---
COL_BG = "#0E0F13"
COL_CARD = "#1A1C22"
COL_CARD2 = "#151821"
COL_PRIMARY = "#5B8CFF"
COL_SUCCESS = "#4ADE80"
COL_WARNING = "#FBBF24"
COL_DANGER = "#FF5C5C"
COL_TEXT = "#F8FAFC"
COL_MUTED = "#94A3B8"


def job_to_document(job: dict) -> Document:
    content = (
        f"תפקיד: {job['title']}\nחברה: {job['company']}\nבכירות: {job['seniority']}\n"
        f"תיאור: {job['description']}\nדרישות: {job['requirements']}"
    )
    return Document(page_content=content, metadata={**job})


@st.cache_resource(show_spinner="בונה אינדקס סמנטי...")
def build_vectorstore() -> Chroma:
    jobs = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
    docs = [job_to_document(j) for j in jobs]
    embeddings = FastEmbedEmbeddings()
    return Chroma.from_documents(documents=docs, embedding=embeddings, collection_name="jobs")


def score_meta(pct: float) -> tuple[str, str]:
    if pct >= 75:
        return COL_SUCCESS, "התאמה גבוהה"
    if pct >= 55:
        return COL_PRIMARY, "התאמה טובה"
    return COL_WARNING, "התאמה חלקית"


def render_results(results: list) -> None:
    """מציג כרטיסי משרה מעוצבים לפי תוצאות החיפוש הסמנטי."""
    st.markdown(
        f"<div style='color:{COL_MUTED};font-size:13px;margin:18px 0 10px'>נמצאו {len(results)} משרות מדורגות לפי התאמה</div>",
        unsafe_allow_html=True,
    )
    for doc, distance in results:
        pct = max(0.0, 1 - distance) * 100
        color, label = score_meta(pct)
        m = doc.metadata
        if m.get("url"):
            apply_url = m["url"]
        else:
            q = urllib.parse.quote(f"{m['title']} {m['company']}")
            apply_url = f"https://www.linkedin.com/jobs/search/?keywords={q}"
        st.markdown(
            f"""
            <div class="jobcard">
                <div class="jc-head">
                    <div>
                        <p class="jc-title">{html.escape(m['title'])}</p>
                        <div class="jc-meta">🏢 {html.escape(m['company'])} &nbsp;·&nbsp; 📍 {html.escape(m['location'])} &nbsp;·&nbsp; {html.escape(m['seniority'])}</div>
                    </div>
                    <div class="scorebox">
                        <div class="num" style="color:{color}">{pct:.0f}%</div>
                        <div class="lbl">{label}</div>
                    </div>
                </div>
                <div class="bar"><span style="width:{min(100,pct):.0f}%;background:{color}"></span></div>
                <div class="jc-body">
                    <b>תיאור:</b> {html.escape(m['description'])}<br>
                    <b>דרישות:</b> {html.escape(m['requirements'])}
                </div>
                <a class="applybtn" href="{html.escape(apply_url)}" target="_blank" rel="noopener">🔗 חפש את המשרה ב-LinkedIn ←</a>
            </div>
            """,
            unsafe_allow_html=True,
        )


def extract_pdf_text(file) -> str:
    """מחלץ טקסט מקובץ PDF שהועלה."""
    from pypdf import PdfReader

    reader = PdfReader(file)
    return "\n".join((p.extract_text() or "") for p in reader.pages)


st.set_page_config(page_title="Career Copilot", page_icon="🎯", layout="centered")

# ============================ עיצוב ============================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', -apple-system, sans-serif !important;
        background: {COL_BG};
    }}
    .stApp {{ background: radial-gradient(1200px 600px at 50% -10%, #16203a 0%, {COL_BG} 55%); }}
    .main .block-container {{ direction: rtl; text-align: right; max-width: 780px; padding-top: 2rem; }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* Hero */
    .hero {{ text-align: center; margin: 8px 0 28px; }}
    .hero .logo {{ font-size: 40px; }}
    .hero h1 {{
        font-size: 34px; font-weight: 700; color: {COL_TEXT};
        margin: 6px 0 4px; letter-spacing: -0.5px;
    }}
    .hero .sub {{ color: {COL_MUTED}; font-size: 15px; font-weight: 400; }}
    .badge-row {{ display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-top:14px; }}
    .pill {{
        background: {COL_CARD}; color: {COL_MUTED}; font-size: 12px; font-weight: 500;
        padding: 5px 12px; border-radius: 999px; border: 1px solid #262a35;
    }}

    /* Inputs */
    .stTextInput input {{
        background: {COL_CARD} !important; color: {COL_TEXT} !important;
        border: 1px solid #2a2f3b !important; border-radius: 14px !important;
        padding: 14px 16px !important; font-size: 15px !important;
    }}
    .stTextInput input:focus {{ border-color: {COL_PRIMARY} !important; box-shadow: 0 0 0 3px rgba(91,140,255,.15) !important; }}
    label, .stSlider label {{ color: {COL_MUTED} !important; font-size: 13px !important; font-weight: 500 !important; }}

    /* Button */
    .stButton > button {{
        background: {COL_PRIMARY}; color: #fff; border: none; border-radius: 14px;
        padding: 12px 20px; font-weight: 600; font-size: 15px; width: 100%;
        transition: transform .18s ease, box-shadow .18s ease, opacity .18s ease;
        box-shadow: 0 6px 20px rgba(91,140,255,.28);
    }}
    .stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 10px 28px rgba(91,140,255,.4); }}

    /* Result card */
    .jobcard {{
        background: {COL_CARD}; border: 1px solid #242833; border-radius: 18px;
        padding: 20px 22px; margin-bottom: 16px;
        box-shadow: 0 8px 30px rgba(0,0,0,.35);
        animation: fadeUp .25s ease both;
    }}
    @keyframes fadeUp {{ from {{ opacity:0; transform: translateY(8px); }} to {{ opacity:1; transform:none; }} }}
    .jc-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:14px; }}
    .jc-title {{ font-size: 19px; font-weight: 600; color: {COL_TEXT}; margin:0; }}
    .jc-meta {{ color: {COL_MUTED}; font-size: 13px; margin-top: 4px; }}
    .scorebox {{ text-align:center; min-width: 72px; }}
    .scorebox .num {{ font-size: 26px; font-weight: 700; line-height:1; }}
    .scorebox .lbl {{ font-size: 11px; color: {COL_MUTED}; margin-top:3px; }}
    .bar {{ height: 6px; background:#23272f; border-radius:999px; margin:14px 0 4px; overflow:hidden; }}
    .bar > span {{ display:block; height:100%; border-radius:999px; }}
    .jc-body {{ color:#c8cedb; font-size:13.5px; line-height:1.6; margin-top:12px; border-top:1px solid #23272f; padding-top:12px; }}
    .jc-body b {{ color:{COL_TEXT}; font-weight:600; }}
    .applybtn {{
        display:inline-block; margin-top:14px; background:rgba(91,140,255,.12);
        color:{COL_PRIMARY}; border:1px solid rgba(91,140,255,.35); border-radius:12px;
        padding:9px 16px; font-size:13px; font-weight:600; text-decoration:none;
        transition: background .18s ease, transform .18s ease;
    }}
    .applybtn:hover {{ background:rgba(91,140,255,.22); transform:translateY(-1px); }}
    .footer {{ text-align:center; color:{COL_MUTED}; font-size:12px; margin-top:30px; }}
    .footer a {{ color:{COL_PRIMARY}; text-decoration:none; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================ Hero ============================
st.markdown(
    """
    <div class="hero">
        <div class="logo">🎯</div>
        <h1>Career Copilot</h1>
        <div class="sub">חיפוש סמנטי במשרות הייטק לפי משמעות — לא מילות מפתח</div>
        <div class="badge-row">
            <span class="pill">RAG</span>
            <span class="pill">Embeddings</span>
            <span class="pill">Vector Search</span>
            <span class="pill">Claude</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["🔍 חיפוש חופשי", "📄 התאמה לפי קורות חיים"])

with tab1:
    query = st.text_input("מה אתה מחפש?", value="משרות AI ואוטומציה עם רקע בתפעול ושירות")
    k1 = st.slider("כמה תוצאות?", 1, 8, 4, key="k1")
    if st.button("🔍 חפש משרות", key="b1") and query.strip():
        vs = build_vectorstore()
        render_results(vs.similarity_search_with_score(query, k=k1))

with tab2:
    st.markdown(
        f"<div style='color:{COL_MUTED};font-size:13px;margin-bottom:10px'>"
        "הדבק את קורות החיים שלך או העלה PDF — והמערכת תדרג את המשרות לפי ההתאמה אליך."
        "</div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("העלה קובץ PDF של קורות חיים", type=["pdf"])
    cv_text = st.text_area("או הדבק את הטקסט של קורות החיים:", height=200,
                           placeholder="שם, ניסיון תעסוקתי, כישורים, השכלה...")
    k2 = st.slider("כמה תוצאות?", 1, 8, 4, key="k2")
    if st.button("📄 התאם משרות לקורות החיים", key="b2"):
        profile_text = ""
        if uploaded is not None:
            with st.spinner("קורא את ה-PDF..."):
                profile_text = extract_pdf_text(uploaded)
        elif cv_text.strip():
            profile_text = cv_text
        if not profile_text.strip():
            st.warning("נא להעלות קובץ PDF או להדביק טקסט של קורות חיים.")
        else:
            vs = build_vectorstore()
            render_results(vs.similarity_search_with_score(profile_text, k=k2))

st.markdown(
    '<div class="footer">נבנה כפרויקט RAG · <a href="https://github.com/bke1302/career-copilot" target="_blank">קוד מקור ב-GitHub</a></div>',
    unsafe_allow_html=True,
)

"""
שלב 3 — דירוג התאמה + התאמת קורות חיים אוטומטית (RAG מלא)
==========================================================
זה ה-RAG השלם. הזרימה:
1. RETRIEVAL  — מטמיעים את הפרופיל ומחזירים מ-Chroma את המשרות הקרובות ביותר
2. GENERATION — לכל משרה, Claude מקבל [הפרופיל + המשרה] ומחזיר:
   • ציון התאמה 0-100 + נימוק
   • חוזקות מול המשרה + פערים
   • תקציר מקצועי מותאם + בולטים מותאמים לקו"ח

הריטריבל רץ מקומית (בחינם). רק הגנרציה עוברת ל-Claude API.

דרישה: משתנה סביבה ANTHROPIC_API_KEY
הרצה:
    python stage3_match.py
"""

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # פלט UTF-8 ל-Windows

import anthropic
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"
PROFILE_FILE = BASE_DIR / "profile.txt"
COLLECTION_NAME = "jobs"

MODEL = "claude-opus-4-8"   # המודל החזק ביותר. אפשר להחליף ל-claude-sonnet-4-6 לזול/מהיר יותר.
TOP_K = 3                   # כמה משרות מובילות לעבד

# מבנה ה-JSON שנכריח את Claude להחזיר — מבטיח פלט עקבי ובר-עיבוד (Structured Outputs)
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "match_score": {"type": "integer", "description": "ציון התאמה 0-100"},
        "verdict": {"type": "string", "description": "שורת סיכום אחת על ההתאמה"},
        "strengths": {"type": "array", "items": {"type": "string"}, "description": "חוזקות הפרופיל מול המשרה"},
        "gaps": {"type": "array", "items": {"type": "string"}, "description": "פערים / מה חסר"},
        "tailored_summary": {"type": "string", "description": "תקציר מקצועי מותאם למשרה הזו"},
        "tailored_bullets": {"type": "array", "items": {"type": "string"}, "description": "3-5 בולטים לקו\"ח מותאמים למשרה"},
    },
    "required": ["match_score", "verdict", "strengths", "gaps", "tailored_summary", "tailored_bullets"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "אתה יועץ קריירה מומחה בהייטק. תפקידך לדרג התאמה בין פרופיל מועמד למשרה, "
    "ולהתאים את קורות החיים למשרה הספציפית. היה כן וענייני לגבי פערים — אל תנפח ציונים. "
    "ענה תמיד בעברית. התבסס רק על המידע בפרופיל; אל תמציא ניסיון שלא קיים."
)


def get_client() -> anthropic.Anthropic:
    # מקור 1: משתנה סביבה. מקור 2 (גיבוי): קובץ api_key.txt בתיקיית הפרויקט.
    # ניקוי אגרסיבי: רווחים, מעברי שורה ותווי BOM נסתרים פוסלים את כותרת ה-HTTP.
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip().strip("﻿").strip()
    key_file = BASE_DIR / "api_key.txt"
    if not key and key_file.exists():
        key = key_file.read_text(encoding="utf-8-sig").strip().strip("﻿").strip()
    if not key:
        print("❌ חסר מפתח API. צור קובץ api_key.txt עם המפתח, או הגדר משתנה סביבה.")
        sys.exit(1)
    return anthropic.Anthropic(api_key=key)


def analyze_job(client: anthropic.Anthropic, profile: str, job_content: str) -> dict:
    """קריאה אחת ל-Claude: מחזירה דירוג + קו\"ח מותאם כ-JSON תקין."""
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
        # Structured Outputs: מכריח את הפלט להיות JSON תואם-סכימה
        output_config={"format": {"type": "json_schema", "schema": ANALYSIS_SCHEMA}},
    )
    # עם output_config.format, בלוק הטקסט מכיל JSON תקין
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def main() -> None:
    profile = PROFILE_FILE.read_text(encoding="utf-8")
    print(f"👤 פרופיל נטען ({len(profile)} תווים)\n")

    # --- שלב RETRIEVAL: אותו מודל embeddings כמו בשלבים 1-2 ---
    embeddings = FastEmbedEmbeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )
    results = vectorstore.similarity_search_with_score(profile, k=TOP_K)
    print(f"🔎 {len(results)} המשרות הרלוונטיות ביותר אותרו. שולח ל-Claude לניתוח...\n")

    # --- שלב GENERATION: ניתוח כל משרה עם Claude ---
    client = get_client()
    for rank, (doc, _distance) in enumerate(results, start=1):
        title = doc.metadata["title"]
        company = doc.metadata["company"]
        print("═" * 60)
        print(f"#{rank}  {title} @ {company}")
        print("═" * 60)

        analysis = analyze_job(client, profile, doc.page_content)

        print(f"🎯 ציון התאמה: {analysis['match_score']}/100")
        print(f"📌 {analysis['verdict']}\n")

        print("✅ חוזקות:")
        for s in analysis["strengths"]:
            print(f"   • {s}")

        print("\n⚠️  פערים:")
        for g in analysis["gaps"]:
            print(f"   • {g}")

        print(f"\n📝 תקציר מקצועי מותאם:\n   {analysis['tailored_summary']}")

        print("\n📄 בולטים מותאמים לקו\"ח:")
        for b in analysis["tailored_bullets"]:
            print(f"   • {b}")
        print()


if __name__ == "__main__":
    main()

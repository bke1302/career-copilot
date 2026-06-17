"""
שלב 1 — טעינת משרות והטמעתן ב-Vector Database (Chroma)
========================================================
מה הסקריפט עושה:
1. טוען את המשרות מקובץ jobs.json
2. הופך כל משרה ל-Document (טקסט לחיפוש + מטא-דאטה מובנה)
3. מחשב embedding לכל משרה (וקטור מספרים שמייצג משמעות)
4. שומר את הכל ב-Chroma על הדיסק, בתיקייה chroma_db/
"""

import json
import shutil
import sys
from pathlib import Path

# Windows: מכריחים פלט UTF-8 כדי שעברית ואמוג'י לא יקרסו ב-console
sys.stdout.reconfigure(encoding="utf-8")

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# נתיבים — הכל יחסית לתיקיית הסקריפט, כדי שירוץ מכל מקום
BASE_DIR = Path(__file__).parent
JOBS_FILE = BASE_DIR / "jobs.json"
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "jobs"


def load_jobs() -> list[dict]:
    """קורא את קובץ ה-JSON ומחזיר רשימת מילונים (כל מילון = משרה)."""
    with open(JOBS_FILE, encoding="utf-8") as f:
        return json.load(f)


def job_to_document(job: dict) -> Document:
    """
    הופך משרה ל-Document של LangChain.

    page_content = הטקסט שעליו מחושב ה-embedding וייעשה עליו חיפוש סמנטי.
                   מאחדים את כל השדות החשובים לטקסט אחד עשיר.
    metadata     = שדות מובנים שנשמרים לצד הוקטור (לסינון/תצוגה בהמשך).
    """
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
            "id": job["id"],
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "seniority": job["seniority"],
        },
    )


def main() -> None:
    # אם כבר קיים אינדקס קודם — מוחקים ובונים מאפס (כדי למנוע כפילויות בהרצה חוזרת)
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
        print(f"🗑️  אינדקס קודם נמחק ({DB_DIR.name})")

    jobs = load_jobs()
    print(f"📄 נטענו {len(jobs)} משרות מ-{JOBS_FILE.name}")

    documents = [job_to_document(j) for j in jobs]
    ids = [j["id"] for j in jobs]

    # מנוע ה-embeddings המקומי. בהרצה הראשונה הוא מוריד מודל קטן (~90MB) ושומר במטמון.
    print("🧠 טוען מודל embeddings מקומי (הורדה חד-פעמית בפעם הראשונה)...")
    embeddings = FastEmbedEmbeddings()

    # יוצרים את ה-vector store, מחשבים embedding לכל משרה ושומרים לדיסק.
    print("💾 מחשב embeddings ושומר ב-Chroma...")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
    )

    print(f"✅ הושלם! {len(documents)} משרות הוטמעו ונשמרו ב-{DB_DIR.name}/")


if __name__ == "__main__":
    main()

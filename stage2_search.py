"""
שלב 2 — חיפוש סמנטי (Semantic Search)
======================================
מה הסקריפט עושה:
1. טוען את אינדקס Chroma שנבנה בשלב 1 (לא בונה מחדש!)
2. ממיר את השאלה שלך ל-embedding (אותו מודל כמו בשלב 1 — חובה!)
3. מחפש את המשרות שהוקטור שלהן הכי קרוב לוקטור השאלה
4. מדפיס את התוצאות לפי סדר התאמה

הקסם: החיפוש מבוסס *משמעות*, לא מילים. שאלה על "רקע בתפעול"
תמצא משרה שכתוב בה "ניסיון בשירות" — גם בלי מילה משותפת.

הרצה:
    python stage2_search.py "השאלה שלך כאן"
    python stage2_search.py            # משתמש בשאלת ברירת מחדל
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # פלט UTF-8 ל-Windows

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "jobs"

# כמה תוצאות להחזיר
TOP_K = 4


def main() -> None:
    # השאלה מגיעה מה-command line, או ברירת מחדל אם לא סופקה
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "משרות AI ואוטומציה עם רקע בתפעול ושירות"

    print(f"🔎 שאלה: {query}\n")

    # חשוב: אותו מודל embeddings כמו בשלב 1, אחרת הוקטורים לא ישתוו
    embeddings = FastEmbedEmbeddings()

    # טוענים את האינדקס הקיים מהדיסק (לא from_documents — אנחנו רק קוראים)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )

    # החיפוש עצמו. מחזיר זוגות (משרה, score).
    # ב-Chroma ה-score הוא "מרחק" — נמוך יותר = קרוב/דומה יותר.
    results = vectorstore.similarity_search_with_score(query, k=TOP_K)

    print(f"📋 {len(results)} המשרות הרלוונטיות ביותר:\n")
    for rank, (doc, distance) in enumerate(results, start=1):
        # ממירים מרחק לאחוז "דמיון" אינטואיטיבי (ככל שהמרחק קטן, הדמיון גבוה)
        similarity = max(0.0, 1 - distance) * 100
        print(f"#{rank}  [{similarity:.0f}% התאמה]  {doc.metadata['title']}")
        print(f"     🏢 {doc.metadata['company']}  |  📍 {doc.metadata['location']}")
        print()


if __name__ == "__main__":
    main()

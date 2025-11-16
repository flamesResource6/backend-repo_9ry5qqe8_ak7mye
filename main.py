import os
import re
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# PDF/DOCX parsing libs
try:
    from PyPDF2 import PdfReader
except Exception:  # library may be missing until requirements are installed
    PdfReader = None  # type: ignore
try:
    import docx  # python-docx
except Exception:
    docx = None  # type: ignore

app = FastAPI(title="ATS Resume Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisResult(BaseModel):
    score: int
    summary: str
    issues: List[str]
    suggestions: List[str]
    keywords_found: List[str]
    keywords_missing: List[str]
    word_count: int
    estimated_pages: int


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db  # type: ignore

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# ----------- ATS ANALYZER -----------
SECTION_KEYWORDS = {
    "summary": ["summary", "objective", "profile"],
    "experience": ["experience", "work history", "employment"],
    "education": ["education", "qualifications", "degree"],
    "skills": ["skills", "technical skills", "tooling"],
    "projects": ["projects", "portfolio"],
    "certifications": ["certifications", "certificates", "licenses"],
    "contact": ["email", "phone", "linkedin", "github"],
}

COMMON_SKILLS = [
    # General
    "communication", "leadership", "team", "collaboration", "problem-solving",
    # Tech
    "python", "javascript", "react", "node", "java", "sql", "aws", "docker", "kubernetes",
    "machine learning", "data analysis", "git", "typescript", "html", "css",
]


def extract_text_from_upload(file: UploadFile) -> str:
    name = file.filename or "resume"
    ext = os.path.splitext(name)[1].lower()
    content = file.file.read()

    # Safety: limit file size to ~5MB
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 5MB)")

    if ext in [".txt", ""]:
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return content.decode(errors="ignore")

    if ext == ".pdf":
        if PdfReader is None:
            raise HTTPException(status_code=500, detail="PDF parser not available")
        try:
            from io import BytesIO
            reader = PdfReader(BytesIO(content))
            pages_text = []
            for p in reader.pages:
                try:
                    pages_text.append(p.extract_text() or "")
                except Exception:
                    pages_text.append("")
            return "\n".join(pages_text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read PDF: {str(e)[:120]}")

    if ext in [".docx"]:
        if docx is None:
            raise HTTPException(status_code=500, detail="DOCX parser not available")
        try:
            from io import BytesIO
            d = docx.Document(BytesIO(content))
            return "\n".join([p.text for p in d.paragraphs])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read DOCX: {str(e)[:120]}")

    raise HTTPException(status_code=415, detail="Unsupported file type. Upload PDF, DOCX, or TXT.")


def simple_clean(text: str) -> str:
    # Normalize whitespace and lowercase for analysis
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_sections(text: str) -> List[str]:
    found = []
    lower = text.lower()
    for section, keys in SECTION_KEYWORDS.items():
        if any(k in lower for k in keys):
            found.append(section)
    return found


def find_contact_info(text: str) -> List[str]:
    hits = []
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text, re.I):
        hits.append("email")
    if re.search(r"(\+?\d[\d\s().-]{7,})", text):
        hits.append("phone")
    if re.search(r"linkedin\.com/", text, re.I):
        hits.append("linkedin")
    if re.search(r"github\.com/", text, re.I):
        hits.append("github")
    return hits


def keyword_match(text: str, job_description: Optional[str]) -> (List[str], List[str]):
    if job_description:
        jd = job_description.lower()
        # naive extract of keywords: words longer than 3 letters, dedup
        words = set(re.findall(r"[a-zA-Z][a-zA-Z+.#/-]{3,}", jd))
        # Filter very common stop words roughly
        stop = {"with","have","this","that","from","will","your","you","they","them","into","such","including","using","able","must","shall","also"}
        keywords = sorted([w for w in words if w.lower() not in stop])[:30]
    else:
        keywords = COMMON_SKILLS

    text_lower = text.lower()
    found = [k for k in keywords if k.lower() in text_lower]
    missing = [k for k in keywords if k.lower() not in text_lower]
    return found, missing


def score_resume(text: str, job_description: Optional[str]) -> AnalysisResult:
    cleaned = simple_clean(text)
    word_count = max(1, len(cleaned.split()))

    # Base score and deductions
    score = 100
    issues: List[str] = []
    suggestions: List[str] = []

    # Sections
    sections_found = detect_sections(cleaned)
    essential = {"summary", "experience", "education", "skills"}
    missing_sections = [s for s in essential if s not in sections_found]
    if missing_sections:
        issues.append(f"Missing sections: {', '.join(missing_sections)}")
        score -= 10
        for s in missing_sections:
            suggestions.append(f"Add a clear '{s.title()}' section with a heading.")

    # Contact info
    contact = find_contact_info(cleaned)
    if not contact:
        issues.append("Contact details not detected")
        suggestions.append("Include email, phone and a LinkedIn URL in the header.")
        score -= 10

    # Length heuristic
    estimated_pages = 1 + (word_count // 700)
    if estimated_pages > 2:
        issues.append("Resume appears longer than 2 pages")
        suggestions.append("Keep it concise: target 1 page for <8 years experience, otherwise 2.")
        score -= 10

    # Bullet points
    bullets = len(re.findall(r"^\s*[•\-\u2022]", text, re.M)) + len(re.findall(r"\n- ", text))
    if bullets < 5:
        issues.append("Very few bullet points detected")
        suggestions.append("Use bullet points to highlight achievements and impact.")
        score -= 5

    # All-caps words (may indicate ATS issues if excessive)
    caps_words = re.findall(r"\b[A-Z]{4,}\b", text)
    if len(caps_words) > 40:
        issues.append("Excessive ALL-CAPS detected")
        suggestions.append("Use Title Case; avoid overusing ALL CAPS which may hurt readability.")
        score -= 5

    # Keywords vs JD or common skills
    found, missing = keyword_match(cleaned, job_description)
    if missing:
        issues.append("Some important keywords are missing")
        suggestions.append("Incorporate relevant keywords naturally in your experience and skills.")
        score -= min(20, len(missing) // 3 * 3)  # mild penalty

    # Cap score between 0 and 100
    score = max(0, min(100, score))

    summary = (
        "Strong structure with key sections and contact info." if not issues else
        "We found areas to improve for better ATS compatibility."
    )

    return AnalysisResult(
        score=score,
        summary=summary,
        issues=issues,
        suggestions=suggestions,
        keywords_found=found,
        keywords_missing=missing[:20],
        word_count=word_count,
        estimated_pages=estimated_pages,
    )


@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_resume(
    file: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
):
    """Upload a resume (PDF, DOCX, or TXT) and get an ATS score + suggestions.
    Optionally pass a job_description to tailor keyword matching.
    """
    text = extract_text_from_upload(file)
    result = score_resume(text, job_description)
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

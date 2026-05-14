"""
FastAPI Backend v2 for the Hallucination Detector Chrome Extension.
Run with:  python -m uvicorn app:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our Shadow AI engine (model loads on first import)
from shadow_detector import full_analysis, analyze_conflict

app = FastAPI(
    title="Shadow AI - Hallucination Detector v2",
    description="Calibrated hallucination analysis using Z-Score entropy and multi-factor scoring.",
    version="2.0.0",
)

# Allow the Chrome extension to call our API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response Models ---
class AnalysisRequest(BaseModel):
    prompt: str
    response: str


# --- Endpoints ---
@app.get("/health")
async def health_check():
    """Check if the Shadow AI server is running and the model is loaded."""
    return {
        "status": "online",
        "model": "Qwen1.5-0.5B-Chat",
        "version": "2.0.0",
        "message": "Shadow AI v2 (Calibrated) is ready.",
    }


@app.post("/analyze")
async def analyze(data: AnalysisRequest):
    """
    Full calibrated analysis. Uses Z-Score entropy, multi-factor conflict scoring,
    and reprompt suggestion engine. Returns overall score, per-token data, and suggestions.
    """
    report = full_analysis(data.prompt, data.response)
    return report


@app.post("/quick-check")
async def quick_check(data: AnalysisRequest):
    """
    Lightweight conflict-only check. Use for real-time streaming where speed matters.
    """
    conflict_report = analyze_conflict(data.prompt, data.response)

    high_conflict = sum(1 for t in conflict_report if t["status"] == "HIGH_CONFLICT")
    warn = sum(1 for t in conflict_report if t["status"] == "WARN")

    return {
        "total_tokens": len(conflict_report),
        "high_conflict": high_conflict,
        "warnings": warn,
        "flagged_tokens": [
            {"token": t["token"], "conflict": t["conflict_score"], "shadow_preferred": t["shadow_preferred"]}
            for t in conflict_report
            if t["status"] in ("HIGH_CONFLICT", "WARN")
        ],
    }

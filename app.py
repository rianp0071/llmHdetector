"""
FastAPI Backend for the Hallucination Detector Chrome Extension.
Run with:  uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our Shadow AI engine (model loads on first import)
from shadow_detector import full_analysis, analyze_conflict, analyze_existing_response

app = FastAPI(
    title="Shadow AI - Hallucination Detector",
    description="Analyzes AI-generated text for hallucinations using mechanistic interpretability.",
    version="1.0.0",
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


class HealthResponse(BaseModel):
    status: str
    model: str
    message: str


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the Shadow AI server is running and the model is loaded."""
    return {
        "status": "online",
        "model": "Qwen1.5-0.5B-Chat",
        "message": "Shadow AI is ready to analyze.",
    }


@app.post("/analyze")
async def analyze(data: AnalysisRequest):
    """
    Main endpoint. Runs the full hallucination analysis pipeline:
    - Attention-based influence tracking
    - Entropy (confusion) measurement
    - Conflict scoring (Shadow vs. External AI)
    - Logit divergence detection
    Returns a combined report with an overall hallucination score (0-100).
    """
    report = full_analysis(data.prompt, data.response)
    return report


@app.post("/quick-check")
async def quick_check(data: AnalysisRequest):
    """
    Lightweight endpoint that only returns the conflict scores per token.
    Use this for real-time streaming analysis where speed matters.
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

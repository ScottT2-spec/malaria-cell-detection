"""
MalariaAI — FastAPI Backend
CNN Detection + K2 Think V2 Clinical Intelligence + RAG + Doctor Feedback
"""

import os
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

from k2_think import K2ThinkClient
from rag.engine import RAGEngine
from feedback.store import CorrectionStore
from feedback.corrections import process_correction
from reports.generator import ReportGenerator
from ml.retrainer import ModelRetrainer

# ── Config ──────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GUIDELINES_DIR = DATA_DIR / "guidelines"
CORRECTIONS_DIR = DATA_DIR / "corrections"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

K2_API_KEY = os.getenv("K2_API_KEY", "")
K2_API_URL = os.getenv("K2_API_URL", "https://api.k2think.ai/v1/chat/completions")
K2_MODEL = os.getenv("K2_MODEL", "MBZUAI-IFM/K2-Think-v2")

# ── App Lifespan ────────────────────────────────────────────────

rag_engine: RAGEngine = None
correction_store: CorrectionStore = None
k2_client: K2ThinkClient = None
report_generator: ReportGenerator = None
model_retrainer: ModelRetrainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine, correction_store, k2_client, report_generator

    # Initialize RAG engine with guidelines + corrections
    rag_engine = RAGEngine(
        guidelines_dir=str(GUIDELINES_DIR),
        corrections_dir=str(CORRECTIONS_DIR),
        embeddings_dir=str(EMBEDDINGS_DIR),
    )
    rag_engine.load()

    # Initialize correction store
    correction_store = CorrectionStore(str(CORRECTIONS_DIR))

    # Initialize K2 Think client
    k2_client = K2ThinkClient(
        api_key=K2_API_KEY,
        api_url=K2_API_URL,
        model=K2_MODEL,
    )

    # Initialize report generator
    report_generator = ReportGenerator(k2_client=k2_client)

    # Initialize model retrainer
    model_retrainer = ModelRetrainer(
        model_dir=str(BASE_DIR / "model"),
        corrections_dir=str(CORRECTIONS_DIR),
        rehearsal_dir=str(DATA_DIR / "rehearsal"),
        history_dir=str(DATA_DIR / "retrain_history"),
    )

    print("✅ MalariaAI backend ready")
    print(f"   📚 RAG: {rag_engine.total_chunks} chunks loaded")
    print(f"   💊 Guidelines: {rag_engine.guideline_count} documents")
    print(f"   🩺 Corrections: {correction_store.count} on file")
    print(f"   🧬 Retrainer: {model_retrainer.correction_count()['total']} correction images")

    yield

    print("🛑 MalariaAI backend shutting down")


# ── FastAPI App ─────────────────────────────────────────────────

app = FastAPI(
    title="MalariaAI",
    description="AI-powered malaria screening with K2 Think V2 clinical intelligence",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files and model
FRONTEND_DIR = BASE_DIR / "frontend"
DOCS_DIR = BASE_DIR / "docs"


# ── Request / Response Models ───────────────────────────────────


class AnalysisRequest(BaseModel):
    """Sent by frontend after CNN runs in-browser."""
    scan_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    prediction: str  # "parasitized" | "uninfected"
    confidence: float  # 0.0 – 1.0
    infected_prob: float
    healthy_prob: float
    inference_time_ms: int
    image_hash: Optional[str] = None
    patient_id: Optional[str] = None


class AnalysisResponse(BaseModel):
    scan_id: str
    prediction: str
    confidence: float
    clinical_summary: str
    treatment_guidance: str
    patient_education: str
    severity_assessment: str
    guidelines_cited: list[str]
    follow_up_questions: list[str]
    rag_sources_used: int
    timestamp: str


class CorrectionRequest(BaseModel):
    scan_id: str
    original_prediction: str
    corrected_prediction: str
    original_confidence: Optional[float] = None
    corrected_species: Optional[str] = None  # P. falciparum, P. vivax, etc.
    parasitemia_level: Optional[str] = None  # low, moderate, high
    doctor_notes: Optional[str] = None
    doctor_id: Optional[str] = None


class CorrectionResponse(BaseModel):
    correction_id: str
    status: str
    message: str
    rag_updated: bool


class ChatRequest(BaseModel):
    scan_id: Optional[str] = None
    message: str
    conversation_history: list[dict] = []


class ChatResponse(BaseModel):
    response: str
    sources_cited: list[str]


class ReportRequest(BaseModel):
    scan_id: str
    patient_id: Optional[str] = None
    analysis: dict
    include_guidelines: bool = True


# ── Routes ──────────────────────────────────────────────────────


@app.get("/")
async def root():
    """Serve frontend."""
    index_path = FRONTEND_DIR / "templates" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"status": "ok", "service": "MalariaAI", "version": "2.0.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "rag_chunks": rag_engine.total_chunks if rag_engine else 0,
        "corrections": correction_store.count if correction_store else 0,
        "k2_configured": bool(K2_API_KEY),
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """
    Main analysis endpoint.
    Frontend sends CNN results → backend adds K2 Think clinical intelligence via RAG.
    """
    # 1. Smart RAG retrieval — builds multiple queries based on CNN results
    rag_context = rag_engine.retrieve_for_analysis(
        prediction=request.prediction,
        confidence=request.confidence,
    )

    # 2. Find relevant doctor corrections (matched by prediction + confidence range)
    similar_corrections = correction_store.find_similar(
        prediction=request.prediction,
        confidence=request.confidence,
    )

    # 3. Call K2 Think V2 for clinical intelligence
    k2_response = await k2_client.clinical_analysis(
        prediction=request.prediction,
        confidence=request.confidence,
        infected_prob=request.infected_prob,
        healthy_prob=request.healthy_prob,
        rag_context=rag_context,
        corrections_context=similar_corrections,
    )

    return AnalysisResponse(
        scan_id=request.scan_id,
        prediction=request.prediction,
        confidence=request.confidence,
        clinical_summary=k2_response["clinical_summary"],
        treatment_guidance=k2_response["treatment_guidance"],
        patient_education=k2_response["patient_education"],
        severity_assessment=k2_response["severity_assessment"],
        guidelines_cited=k2_response["guidelines_cited"],
        follow_up_questions=k2_response["follow_up_questions"],
        rag_sources_used=len(rag_context),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/correct", response_model=CorrectionResponse)
async def submit_correction(request: CorrectionRequest):
    """
    Doctor correction endpoint.
    Saves correction and updates RAG knowledge base in real-time.
    """
    # Save correction to store
    correction_id = correction_store.save(
        scan_id=request.scan_id,
        original=request.original_prediction,
        corrected=request.corrected_prediction,
        species=request.corrected_species,
        parasitemia=request.parasitemia_level,
        notes=request.doctor_notes,
        doctor_id=request.doctor_id,
        confidence=request.original_confidence,
    )

    # Update RAG engine with new correction (live learning — immediate)
    rag_updated = rag_engine.add_correction(
        correction_id=correction_id,
        text=process_correction(request),
    )

    return CorrectionResponse(
        correction_id=correction_id,
        status="saved",
        message="Correction recorded. The AI will incorporate this in future analyses.",
        rag_updated=rag_updated,
    )


@app.post("/api/correct/image")
async def submit_correction_with_image(
    scan_id: str = Form(...),
    original_prediction: str = Form(...),
    corrected_prediction: str = Form(...),
    original_confidence: float = Form(0.0),
    corrected_species: Optional[str] = Form(None),
    parasitemia_level: Optional[str] = Form(None),
    doctor_notes: Optional[str] = Form(None),
    doctor_id: Optional[str] = Form(None),
    image: UploadFile = File(...),
):
    """
    Doctor correction WITH image — enables CNN retraining.
    The image is stored for future model fine-tuning.
    The correction text is added to RAG immediately.
    """
    # Read image bytes
    image_bytes = await image.read()

    # Save correction to text store + RAG (immediate learning)
    correction_id = correction_store.save(
        scan_id=scan_id,
        original=original_prediction,
        corrected=corrected_prediction,
        species=corrected_species,
        parasitemia=parasitemia_level,
        notes=doctor_notes,
        doctor_id=doctor_id,
        confidence=original_confidence,
    )

    # Build correction request for RAG text
    class _Req:
        pass
    req = _Req()
    req.scan_id = scan_id
    req.original_prediction = original_prediction
    req.corrected_prediction = corrected_prediction
    req.corrected_species = corrected_species
    req.parasitemia_level = parasitemia_level
    req.doctor_notes = doctor_notes

    rag_updated = rag_engine.add_correction(
        correction_id=correction_id,
        text=process_correction(req),
    )

    # Save image for CNN retraining (deferred learning)
    image_path = model_retrainer.save_correction_image(
        scan_id=scan_id,
        image_bytes=image_bytes,
        corrected_label=corrected_prediction,
        original_label=original_prediction,
        confidence=original_confidence,
        metadata={
            "species": corrected_species,
            "parasitemia": parasitemia_level,
            "doctor_notes": doctor_notes,
        },
    )

    counts = model_retrainer.correction_count()

    return {
        "correction_id": correction_id,
        "status": "saved",
        "rag_updated": rag_updated,
        "image_saved": True,
        "image_path": image_path,
        "message": "Correction + image saved. RAG updated immediately. Image stored for CNN retraining.",
        "retrain_status": model_retrainer.can_retrain(),
        "correction_images": counts,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Interactive Q&A — ask follow-up questions about diagnosis, treatment, malaria.
    """
    # RAG retrieval based on user question
    rag_context = rag_engine.retrieve(query=request.message, top_k=3)

    response = await k2_client.chat(
        message=request.message,
        conversation_history=request.conversation_history,
        rag_context=rag_context,
    )

    return ChatResponse(
        response=response["answer"],
        sources_cited=response.get("sources", []),
    )


@app.post("/api/report")
async def generate_report(request: ReportRequest):
    """Generate a professional clinical report using K2 Think."""
    report = await report_generator.generate(
        scan_id=request.scan_id,
        patient_id=request.patient_id,
        analysis=request.analysis,
        include_guidelines=request.include_guidelines,
    )
    return report


@app.get("/api/corrections/stats")
async def correction_stats():
    """Dashboard stats for the feedback system."""
    return correction_store.stats()


@app.get("/api/rag/status")
async def rag_status():
    """RAG engine status."""
    return {
        "total_chunks": rag_engine.total_chunks,
        "guideline_documents": rag_engine.guideline_count,
        "correction_entries": rag_engine.correction_count,
        "last_updated": rag_engine.last_updated,
    }


# ── Model Retraining Endpoints ──────────────────────────────────


@app.get("/api/retrain/status")
async def retrain_status():
    """Check if CNN retraining is possible and how many corrections are queued."""
    return model_retrainer.can_retrain()


@app.post("/api/retrain/run")
async def retrain_model():
    """
    Trigger CNN retraining with accumulated doctor corrections.

    Safety guarantees:
    - Conv layers frozen (only dense head fine-tuned)
    - Mixed batches with rehearsal data (experience replay)
    - EWC penalty prevents catastrophic forgetting
    - Validation gate: new model must beat old model or it's rejected
    - Old model always backed up before replacement
    """
    result = await model_retrainer.retrain()
    return result


@app.get("/api/retrain/history")
async def retrain_history():
    """Get history of all retrain attempts (accepted and rejected)."""
    return model_retrainer.retrain_history()


# ── Static File Mounts (MUST be after all API routes) ───────────

if (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")

if (DOCS_DIR / "model").exists():
    app.mount("/model", StaticFiles(directory=str(DOCS_DIR / "model")), name="model")

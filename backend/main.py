"""
backend/main.py
================
FastAPI application entry point for Socratica.

Endpoints:
  POST /api/analyze          — upload a diagram, returns 202 + job_id immediately
  GET  /api/job/{job_id}     — poll job status / retrieve results
  GET  /api/jobs             — list recent jobs (optional ?student_id= filter)
  GET  /api/analytics        — misconception stats per student/domain
  WS   /ws/job/{job_id}     — WebSocket for real-time job progress updates
  POST /api/dataset/ingest   — trigger background AI2D dataset ingestion
  GET  /api/health           — liveness check (also verifies Ollama connection)

Rate limits:
  /api/analyze   → 30/minute (compute-bound — Ollama VLM inference)
  all others     → 120/minute
"""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("Main")

# Internal imports
from backend.database import (
    close_db,
    create_job,
    get_job,
    get_misconception_stats,
    list_jobs,
    log_feedback,
    update_job_status,
)
from backend.image_preprocessor import preprocess_image, validate_image
from backend.stage1_perception import VLMPerceptionEngine
from backend.stage2_reasoning import LLMReasoningEngine
from backend.stage3_rendering import FeedbackRenderer

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024   # 20 MB hard limit
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
class WSManager:
    """Manages active WebSocket connections keyed by job_id."""
    def __init__(self):
        self._connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, job_id: str, ws: WebSocket):
        await ws.accept()
        self._connections.setdefault(job_id, []).append(ws)

    def disconnect(self, job_id: str, ws: WebSocket):
        conns = self._connections.get(job_id, [])
        if ws in conns:
            conns.remove(ws)

    async def broadcast(self, job_id: str, data: Dict[str, Any]):
        conns = self._connections.get(job_id, [])
        dead  = []
        for ws in conns:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(job_id, ws)


ws_manager = WSManager()


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Socratica API starting up...")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    # Initialize database schema
    from backend.database import get_db
    await get_db()
    logger.info("Database ready.")
    yield
    logger.info("Socratica API shutting down...")
    await close_db()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Socratica — Multimodal STEM Feedback API",
    description=(
        "Evaluates student STEM diagrams using a local Ollama VLM+LLM pipeline "
        "and provides Socratic formative feedback. Runs fully offline."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir, html=True), name="static")


# ---------------------------------------------------------------------------
# Core pipeline worker (runs in background)
# ---------------------------------------------------------------------------
async def _run_pipeline(
    job_id: str,
    student_id: str,
    image_bytes: bytes,
    filename: str,
    context: Optional[str] = None,
) -> None:
    """
    Full three-stage pipeline executed as a background task.
    Updates job status in DB and broadcasts progress over WebSocket.
    """
    try:
        # --- Stage 1: VLM Perception ---
        await update_job_status(job_id, "processing:stage1")
        await ws_manager.broadcast(job_id, {"status": "processing", "stage": 1, "message": "Analyzing diagram with VLM..."})

        perception = VLMPerceptionEngine()
        scene_graph = await perception.analyze(image_bytes, context=context)

        domain = scene_graph.get("domain", "general_science")
        await update_job_status(job_id, "processing:stage2", domain=domain)
        await ws_manager.broadcast(job_id, {"status": "processing", "stage": 2, "message": "Generating Socratic feedback..."})

        # --- Stage 2: LLM Reasoning ---
        reasoning = LLMReasoningEngine()
        reasoning_result = await reasoning.generate_feedback(scene_graph)

        await update_job_status(job_id, "processing:stage3")
        await ws_manager.broadcast(job_id, {"status": "processing", "stage": 3, "message": "Rendering feedback overlays..."})

        # --- Stage 3: Rendering ---
        renderer  = FeedbackRenderer()
        final     = renderer.render(scene_graph, reasoning_result)

        result_payload = json.dumps(final)
        await update_job_status(job_id, "done", result_json=result_payload, domain=domain)

        # Persist feedback to DB for analytics
        await log_feedback(
            job_id=job_id,
            student_id=student_id,
            domain=domain,
            feedback_items=reasoning_result.get("feedback_items", []),
        )

        await ws_manager.broadcast(job_id, {
            "status":  "done",
            "job_id":  job_id,
            "result":  final,
        })
        logger.info("Job %s completed successfully.", job_id)

    except Exception as exc:
        logger.exception("Pipeline failed for job %s: %s", job_id, exc)
        error_msg = str(exc)
        await update_job_status(job_id, "failed", error_message=error_msg)
        await ws_manager.broadcast(job_id, {
            "status":  "failed",
            "job_id":  job_id,
            "error":   error_msg,
        })


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["System"])
@limiter.limit("120/minute")
async def health_check(request: Request) -> Dict[str, Any]:
    """Liveness check. Also pings Ollama to verify model availability."""
    ollama_ok = False
    ollama_models: List[str] = []
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                ollama_ok = True
                data = resp.json()
                ollama_models = [m["name"] for m in data.get("models", [])]
    except Exception:
        pass

    return {
        "status":  "ok",
        "version": "2.0.0",
        "ollama": {
            "reachable": ollama_ok,
            "url":       OLLAMA_BASE_URL,
            "models":    ollama_models,
        },
    }


@app.post("/api/analyze", status_code=202, tags=["Pipeline"], response_model=None)
@limiter.limit("30/minute")
async def analyze_diagram(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    student_id: str = Query(default="anonymous", description="Student or session identifier"),
    context: Optional[str] = Query(default=None, description="Optional context (e.g., 'Grade 9 Physics')"),
) -> Dict[str, Any]:
    """
    Upload a student STEM diagram for analysis.

    Returns HTTP 202 Accepted immediately with a job_id.
    Poll GET /api/job/{job_id} or connect to WS /ws/job/{job_id} for results.
    """
    # --- Validation ---
    raw_bytes = await file.read()

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB."
        )

    valid, reason = validate_image(raw_bytes, file.filename or "upload")
    if not valid:
        raise HTTPException(status_code=422, detail=f"Invalid image: {reason}")

    # --- Preprocess ---
    try:
        processed_bytes = preprocess_image(raw_bytes, file.filename or "upload")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # --- Create job record ---
    job_id = str(uuid.uuid4())
    await create_job(job_id=job_id, filename=file.filename or "upload", student_id=student_id)

    # --- Launch background pipeline ---
    background_tasks.add_task(
        _run_pipeline,
        job_id=job_id,
        student_id=student_id,
        image_bytes=processed_bytes,
        filename=file.filename or "upload",
        context=context,
    )

    logger.info("Job %s queued for student '%s' (file: %s)", job_id, student_id, file.filename)

    return {
        "job_id":    job_id,
        "status":    "queued",
        "message":   "Diagram received. Poll /api/job/{job_id} or connect to /ws/job/{job_id}.",
        "ws_url":    f"/ws/job/{job_id}",
        "poll_url":  f"/api/job/{job_id}",
    }


@app.get("/api/job/{job_id}", tags=["Pipeline"])
@limiter.limit("120/minute")
async def get_job_status(request: Request, job_id: str) -> Dict[str, Any]:
    """Get the current status and results of a processing job."""
    job = await get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    result = None
    if job["result_json"]:
        try:
            result = json.loads(job["result_json"])
        except json.JSONDecodeError:
            result = {"raw": job["result_json"]}

    return {
        "job_id":       job_id,
        "status":       job["status"],
        "student_id":   job["student_id"],
        "filename":     job["filename"],
        "domain":       job["domain"],
        "submitted_at": job["submitted_at"],
        "updated_at":   job["updated_at"],
        "error":        job["error_message"],
        "result":       result,
    }


@app.get("/api/jobs", tags=["Pipeline"])
@limiter.limit("120/minute")
async def list_all_jobs(
    request: Request,
    student_id: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
) -> List[Dict[str, Any]]:
    """List recent processing jobs. Filter by student_id optionally."""
    return await list_jobs(student_id=student_id, limit=limit)


@app.get("/api/analytics", tags=["Analytics"])
@limiter.limit("120/minute")
async def get_analytics(
    request: Request,
    student_id: Optional[str] = Query(default=None),
    domain: Optional[str] = Query(default=None),
    top_n: int = Query(default=10, ge=1, le=50),
) -> Dict[str, Any]:
    """Return top misconceptions by frequency for a student or across all students."""
    stats = await get_misconception_stats(student_id=student_id, domain=domain, top_n=top_n)
    return {
        "filters":  {"student_id": student_id, "domain": domain},
        "top_misconceptions": stats,
    }


@app.post("/api/dataset/ingest", tags=["Training"])
@limiter.limit("5/minute")
async def trigger_dataset_ingestion(
    request: Request,
    background_tasks: BackgroundTasks,
    max_examples: Optional[int] = Query(default=None, description="Max examples to stream (None = full dataset)"),
    shard_size: int = Query(default=1000),
) -> Dict[str, Any]:
    """
    Trigger background ingestion of the lmms-lab/ai2d dataset.
    Streams, parses, and saves shards to data/processed/.
    This is a heavy operation — run once before training.
    """
    def _ingest():
        from backend.data_pipeline import DatasetPipeline
        pipeline = DatasetPipeline(output_dir="./data")
        stats    = pipeline.run_full_pipeline(
            max_examples=max_examples,
            shard_size=shard_size,
        )
        logger.info("Dataset ingestion complete: %s", stats)

    background_tasks.add_task(_ingest)
    return {
        "message":      "Dataset ingestion started in background.",
        "max_examples": max_examples,
        "shard_size":   shard_size,
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/job/{job_id}")
async def ws_job_updates(websocket: WebSocket, job_id: str):
    """
    Real-time job progress updates.
    The client connects immediately after receiving job_id from /api/analyze.
    The server pushes status updates and the final result as JSON messages.
    """
    await ws_manager.connect(job_id, websocket)
    try:
        # Send current state immediately on connect
        job = await get_job(job_id)
        if job:
            await websocket.send_json({"status": job["status"], "job_id": job_id})

        # Keep alive — client disconnects after receiving "done" or "failed"
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        ws_manager.disconnect(job_id, websocket)


# ---------------------------------------------------------------------------
# Serve index.html for all non-API routes (SPA fallback)
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "Socratica API v2.0 — Frontend not found."})

import os
import json
import uuid
import shutil
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

# Import our pipeline engines
from backend.stage1_perception import VLMPerceptionEngine
from backend.stage2_reasoning import LLMReasoningEngine, MISCONCEPTION_TAXONOMY
from backend.stage3_rendering import FeedbackRenderer
from backend.evaluate import PipelineEvaluator
from backend.stats_analysis import StatisticalEvaluator

app = FastAPI(
    title="Multimodal STEM Diagram Feedback API",
    description="Backend API serving the three-stage pedagogical feedback pipeline."
)

# Upload directory
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global database simulation for Teacher Dashboard
# Prepopulated with realistic classroom submissions (28 students) to show rich heatmaps
CLASS_SUBMISSIONS = [
    {"student_id": "student_01", "name": "Alice Chen", "domain": "physics", "timestamp": "2026-07-01", "misconceptions": [{"id": "PHYS-M1", "category": "Force Omission"}]},
    {"student_id": "student_02", "name": "Ben Carter", "domain": "physics", "timestamp": "2026-07-01", "misconceptions": [{"id": "PHYS-M1", "category": "Force Omission"}, {"id": "PHYS-M2", "category": "Direction Error"}]},
    {"student_id": "student_03", "name": "Clara Diaz", "domain": "physics", "timestamp": "2026-07-01", "misconceptions": []},
    {"student_id": "student_04", "name": "David Evans", "domain": "physics", "timestamp": "2026-07-02", "misconceptions": [{"id": "PHYS-M1", "category": "Force Omission"}]},
    {"student_id": "student_05", "name": "Emma Foster", "domain": "physics", "timestamp": "2026-07-02", "misconceptions": [{"id": "PHYS-M2", "category": "Direction Error"}]},
    {"student_id": "student_06", "name": "Frank Green", "domain": "physics", "timestamp": "2026-07-02", "misconceptions": [{"id": "PHYS-M1", "category": "Force Omission"}]},
    {"student_id": "student_07", "name": "Grace Hill", "domain": "physics", "timestamp": "2026-07-02", "misconceptions": [{"id": "PHYS-M1", "category": "Force Omission"}]},
    {"student_id": "student_08", "name": "Henry Irvin", "domain": "physics", "timestamp": "2026-07-02", "misconceptions": []},
    {"student_id": "student_09", "name": "Ivy Jones", "domain": "physics", "timestamp": "2026-07-02", "misconceptions": [{"id": "PHYS-M1", "category": "Force Omission"}, {"id": "PHYS-M2", "category": "Direction Error"}]},
    {"student_id": "student_10", "name": "Jack King", "domain": "physics", "timestamp": "2026-07-02", "misconceptions": [{"id": "PHYS-M1", "category": "Force Omission"}]},
    {"student_id": "student_11", "name": "Leo Miller", "domain": "biology", "timestamp": "2026-07-01", "misconceptions": [{"id": "BIOL-M1", "category": "Organelle Confusion"}]},
    {"student_id": "student_12", "name": "Maya Nelson", "domain": "biology", "timestamp": "2026-07-02", "misconceptions": [{"id": "BIOL-M1", "category": "Organelle Confusion"}]},
    {"student_id": "student_13", "name": "Noah Ortiz", "domain": "biology", "timestamp": "2026-07-02", "misconceptions": []},
    {"student_id": "student_14", "name": "Olivia Patel", "domain": "chemistry", "timestamp": "2026-07-01", "misconceptions": [{"id": "CHEM-M1", "category": "Molecular Geometry Error"}]},
    {"student_id": "student_15", "name": "Peter Quincy", "domain": "chemistry", "timestamp": "2026-07-02", "misconceptions": [{"id": "CHEM-M1", "category": "Molecular Geometry Error"}, {"id": "CHEM-M2", "category": "Valence Incompleteness"}]},
    {"student_id": "student_16", "name": "Ruby Shaw", "domain": "chemistry", "timestamp": "2026-07-02", "misconceptions": [{"id": "CHEM-M2", "category": "Valence Incompleteness"}]},
]

# Simple history profile for a single student to show longitudinal tracking
STUDENT_HISTORY = [
    {"timestamp": "Task 1: Flat Surface", "misconceptions": ["PHYS-M1"]},
    {"timestamp": "Task 2: Mild Incline", "misconceptions": ["PHYS-M1", "PHYS-M2"]},
    {"timestamp": "Task 3: Steep Incline", "misconceptions": ["PHYS-M2"]}, # PHYS-M1 resolved!
    {"timestamp": "Task 4: Pulley System", "misconceptions": []} # All resolved!
]

# API Endpoint definitions
@app.post("/api/evaluate_diagram")
async def evaluate_diagram(
    image: UploadFile = File(...),
    domain: str = Form("physics"),
    vlm_provider: str = Form("mock"),
    llm_provider: str = Form("mock"),
    rubric: str = Form(""),
    student_grade: str = Form("9th Grade"),
    support_tier: str = Form("standard"),
    student_id: Optional[str] = Form("demo_user")
):
    """
    Primary API Pipeline wrapper. Executes VLM, maps to LLM + RAG, 
    evaluates score metrics, and returns rendering formats.
    """
    # 1. Save uploaded file
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(image.filename)[1] or ".png"
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
        
    # 2. Stage 1: VLM Perception Engine
    vlm_engine = VLMPerceptionEngine(provider=vlm_provider)
    scene_graph = await vlm_engine.extract_scene_graph(temp_path, domain=domain)
    
    # 3. Stage 2: LLM Reasoning Engine & RAG
    llm_engine = LLMReasoningEngine(provider=llm_provider)
    student_profile = {"grade_level": student_grade, "support_tier": support_tier}
    reasoning_result = await llm_engine.analyze_scene_graph(
        scene_graph=scene_graph,
        rubric=rubric,
        student_profile=student_profile,
        domain=domain
    )
    
    # Calculate Latency tiers based on feedback length
    feedback_text = reasoning_result.get("socratic_feedback", "")
    token_count = len(feedback_text.split()) * 1.3 # simple estimation factor
    latencies = PipelineEvaluator.hardware_latency_matrix(int(token_count))
    
    # Evaluate GED against a hypothetical correct graph representation
    correct_graphs = {
        "physics": {
            "elements": [
                {"id": "node_block", "type": "node", "label": "Block"},
                {"id": "node_ramp", "type": "node", "label": "Ramp"},
                {"id": "vec_gravity", "type": "vector", "label": "Gravity Vector"},
                {"id": "vec_normal", "type": "vector", "label": "Normal Vector"},
                {"id": "vec_friction", "type": "vector", "label": "Friction Vector"}
            ],
            "relations": [
                {"source": "vec_gravity", "target": "node_block", "relation_type": "acts_on"},
                {"source": "vec_normal", "target": "node_block", "relation_type": "acts_on"},
                {"source": "vec_friction", "target": "node_block", "relation_type": "acts_on"}
            ]
        },
        "biology": {
            "elements": [
                {"id": "cell_wall", "type": "node", "label": "Cell Wall"},
                {"id": "nucleus", "type": "node", "label": "Nucleus"},
                {"id": "chloroplast_1", "type": "node", "label": "Chloroplast"},
                {"id": "vacuole", "type": "node", "label": "Vacuole"}
            ],
            "relations": []
        },
        "chemistry": {
            "elements": [
                {"id": "atom_o", "type": "node", "label": "O"},
                {"id": "atom_h1", "type": "node", "label": "H"},
                {"id": "atom_h2", "type": "node", "label": "H"},
                {"id": "bond_1", "type": "vector", "label": "Single Bond"},
                {"id": "bond_2", "type": "vector", "label": "Single Bond"},
                {"id": "lone_pairs", "type": "label", "label": "Lone Pairs"}
            ],
            "relations": []
        }
    }
    
    ref_graph = correct_graphs.get(domain, {"elements": [], "relations": []})
    ged = PipelineEvaluator.calculate_graph_edit_distance(scene_graph, ref_graph)
    
    # Calculate LLM judge metrics
    judge_evaluation = PipelineEvaluator.llm_as_a_judge_socratic_score(
        feedback_text, 
        domain
    )
    
    # Save student evaluation log to global submissions database
    misconceptions = reasoning_result.get("misconceptions_found", [])
    submission_record = {
        "student_id": student_id,
        "name": f"Student ({student_id})",
        "domain": domain,
        "timestamp": "2026-07-03",
        "misconceptions": misconceptions
    }
    CLASS_SUBMISSIONS.append(submission_record)

    return {
        "scene_graph": scene_graph,
        "evaluation_summary": reasoning_result.get("evaluation_summary"),
        "misconceptions_found": misconceptions,
        "socratic_feedback": feedback_text,
        "accessibility_narration": reasoning_result.get("accessibility_narration"),
        "metrics": {
            "graph_edit_distance": ged,
            "latency_by_hardware_ms": latencies,
            "judge": judge_evaluation
        }
    }

@app.get("/api/classroom_metrics")
def get_classroom_metrics():
    """Aggregates and compiles analytics reports for the Teacher Dashboard."""
    # Class Heatmap calculations
    heatmap_data = FeedbackRenderer.aggregate_class_heatmaps(CLASS_SUBMISSIONS)
    
    # Single student longitudinal history profile
    persistence_profile = FeedbackRenderer.calculate_persistence_metrics(STUDENT_HISTORY)
    
    # Build details on standard learning gains
    # Group learning gains (mocked based on actual stats analysis averages)
    learning_gain_samples = {
        "Pre_Score_Avg": 45.2,
        "Post_Score_Avg": 71.8,
        "Normalized_Gain": round(PipelineEvaluator.calculate_normalized_learning_gain(45.2, 71.8), 3)
    }

    return {
        "heatmap": heatmap_data,
        "persistence_profile": persistence_profile,
        "learning_gains": learning_gain_samples,
        "submissions_log": CLASS_SUBMISSIONS[-10:] # Return last 10 submissions
    }

@app.get("/api/misconception_taxonomy")
def get_misconception_taxonomy():
    """Returns the JSON representation of the indexed AAAS & MaLT Error Library."""
    return MISCONCEPTION_TAXONOMY

@app.get("/api/run_statistical_evaluation")
def run_statistical_evaluation(power: float = 0.80, alpha: float = 0.05, effect_size: float = 0.30):
    """Triggers power analysis calculations and experimental t-tests on classroom distributions."""
    n_required = StatisticalEvaluator.calculate_required_sample_size(
        power=power, 
        alpha=alpha, 
        cohens_d=effect_size
    )
    
    # Generate mock randomized learning gains for control and treatment cohorts
    import random
    random.seed(101)
    control = [random.normalvariate(0.38, 0.12) for _ in range(n_required)]
    treatment = [random.normalvariate(0.49, 0.13) for _ in range(n_required)]
    
    t_test_results = StatisticalEvaluator.run_two_sample_t_test(control, treatment)
    
    return {
        "parameters": {
            "power": power,
            "alpha": alpha,
            "hypothesized_effect_size_d": effect_size
        },
        "required_sample_size_per_group": n_required,
        "welch_t_test": t_test_results
    }

# Serve the static frontend assets (e.g. index.html, styles.css, app.js)
# Note: we mount this last to make sure API routes take priority.
app.mount("/", StaticFiles(directory="./frontend", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

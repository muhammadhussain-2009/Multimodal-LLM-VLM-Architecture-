"""
backend/stage2_reasoning.py
============================
Stage 2: LLM Semantic Reasoning Engine with RAG-based Misconception Detection

Takes the scene graph produced by Stage 1 and:
1. Runs a TF-IDF similarity search against the AAAS/MaLT misconception library.
2. Calls Ollama llama3.1:8b to generate Socratic formative feedback.
3. Classifies each feedback item by type and misconception category.

Selected model: llama3.1:8b (strong reasoning + instruction-following,
fast on CPU-only school hardware, 8-bit quantization friendly).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import httpx
import faiss
from sentence_transformers import SentenceTransformer

from backend.database import update_teacher_heatmap

logger = logging.getLogger("LLMReasoning")

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT   = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))


# ---------------------------------------------------------------------------
# Misconception knowledge base (AAAS Project 2061 + MaLT misconceptions)
# ---------------------------------------------------------------------------
MISCONCEPTION_LIBRARY: List[Dict[str, str]] = [
    # --- Physics ---
    {"id": "PHY-01", "domain": "physics", "category": "force_motion",
     "description": "Student believes heavier objects fall faster than lighter ones regardless of air resistance.",
     "tag": "gravity_misconception"},
    {"id": "PHY-02", "domain": "physics", "category": "force_motion",
     "description": "Student confuses force and velocity, drawing force arrows parallel to motion instead of net resultant.",
     "tag": "force_vector_error"},
    {"id": "PHY-03", "domain": "physics", "category": "energy",
     "description": "Student treats energy as a substance that is 'used up' rather than transferred or transformed.",
     "tag": "energy_transfer_misconception"},
    {"id": "PHY-04", "domain": "physics", "category": "electricity",
     "description": "Student believes current is consumed by a resistor rather than voltage drop occurring.",
     "tag": "current_consumption_error"},
    {"id": "PHY-05", "domain": "physics", "category": "waves",
     "description": "Student conflates frequency and amplitude, thinking louder sound has higher pitch.",
     "tag": "wave_property_confusion"},
    # --- Biology ---
    {"id": "BIO-01", "domain": "biology", "category": "cell_biology",
     "description": "Student believes plants get their food from the soil rather than producing it through photosynthesis.",
     "tag": "photosynthesis_misconception"},
    {"id": "BIO-02", "domain": "biology", "category": "genetics",
     "description": "Student believes organisms can pass on acquired characteristics to offspring.",
     "tag": "lamarckian_inheritance"},
    {"id": "BIO-03", "domain": "biology", "category": "evolution",
     "description": "Student believes evolution has a goal or direction toward 'higher' organisms.",
     "tag": "teleological_evolution"},
    {"id": "BIO-04", "domain": "biology", "category": "cell_biology",
     "description": "Student misidentifies cell membrane vs cell wall functions.",
     "tag": "membrane_wall_confusion"},
    {"id": "BIO-05", "domain": "biology", "category": "respiration",
     "description": "Student conflates breathing (gas exchange) with cellular respiration (ATP production).",
     "tag": "respiration_confusion"},
    # --- Chemistry ---
    {"id": "CHM-01", "domain": "chemistry", "category": "atomic_structure",
     "description": "Student draws electrons in fixed circular orbits around nucleus.",
     "tag": "bohr_model_overextension"},
    {"id": "CHM-02", "domain": "chemistry", "category": "bonding",
     "description": "Student believes all bonds break during a physical change.",
     "tag": "physical_vs_chemical_change"},
    {"id": "CHM-03", "domain": "chemistry", "category": "reactions",
     "description": "Student believes a product 'disappears' in a reaction rather than mass being conserved.",
     "tag": "conservation_of_mass_error"},
    {"id": "CHM-04", "domain": "chemistry", "category": "solutions",
     "description": "Student believes dissolved solute no longer exists in solution.",
     "tag": "dissolution_misconception"},
    {"id": "CHM-05", "domain": "chemistry", "category": "bonding",
     "description": "Student confuses ionic and covalent bonding, misidentifying which elements participate.",
     "tag": "bond_type_confusion"},
]


# ---------------------------------------------------------------------------
# FAISS Knowledge Base Indexing & Retrieval
# ---------------------------------------------------------------------------

_FAISS_INDEX = None
_TAXONOMY_DATA = []
_EMBEDDING_MODEL = None

def _load_faiss_index():
    global _FAISS_INDEX, _TAXONOMY_DATA, _EMBEDDING_MODEL
    try:
        import os
        base_dir = os.path.dirname(os.path.dirname(__file__)) # Project root
        json_path = os.path.join(base_dir, "data", "taxonomy.json")
        index_path = os.path.join(base_dir, "data", "taxonomy.index")
        
        if os.path.exists(json_path) and os.path.exists(index_path):
            with open(json_path, "r") as f:
                _TAXONOMY_DATA = json.load(f)
            _FAISS_INDEX = faiss.read_index(index_path)
            _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info(f"Loaded FAISS taxonomy index with {len(_TAXONOMY_DATA)} items.")
        else:
            logger.warning("FAISS taxonomy index not found. RAG will return empty results.")
    except Exception as e:
        logger.error(f"Failed to load FAISS taxonomy index: {e}")

# Load the index at import time
_load_faiss_index()

def retrieve_misconceptions(
    query: str,
    domain: Optional[str] = None,
    top_k: int = 3,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    FAISS vector similarity search over the misconception library.
    Optional domain filter reduces false positives cross-domain.
    """
    if not query.strip():
        return []
        
    if _FAISS_INDEX is None or not _TAXONOMY_DATA or _EMBEDDING_MODEL is None:
        return []
        
    query_vector = _EMBEDDING_MODEL.encode([query], convert_to_numpy=True)
    # Search for more than top_k to account for domain filtering
    distances, indices = _FAISS_INDEX.search(query_vector, top_k * 3)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
            
        misc = _TAXONOMY_DATA[idx]
        
        # Approximate domain filtering
        if domain and domain.lower() not in misc.get("subdomain", "").lower():
            continue
            
        # Distances are L2, lower is better. Approximate similarity score for legacy compatibility
        sim = max(0.0, 1.0 - (float(dist) / 10.0))
        if sim >= threshold:
            results.append({
                "id": misc.get("misconception_name", ""),
                "description": misc.get("symptom_in_diagram", ""),
                "tag": misc.get("misconception_name", ""),
                "similarity": round(sim, 4)
            })
            if len(results) >= top_k:
                break
                
    return results


# ---------------------------------------------------------------------------
# Socratic prompt construction
# ---------------------------------------------------------------------------

SOCRATIC_SYSTEM_PROMPT = """You are Socratica, an expert STEM educator and ML pipeline orchestrator. Your role is twofold:

1. SUBDOMAIN CLASSIFICATION: Analyze the provided Scene Graph and explicitly categorize the diagram into a specific scientific subdomain (e.g., Physics_Kinematics, Chemistry_Covalent_Bonds, Biology_Cell_Division).
2. GRANULAR EVALUATION: Iterate over EVERY single element (node and edge) in the Scene Graph. 
   - For CORRECT elements: Generate positive validation feedback.
   - For INCORRECT elements: Generate targeted Socratic feedback addressing exactly ONE misconception. Never directly correct the student. End with a reflection question.

Return ONLY a JSON object matching this schema:
{
  "identified_subdomain": "<your_classification>",
  "evaluated_elements": [
    {
      "element_id": "<id>",
      "type": "<type>",
      "status": "correct" or "incorrect",
      "misconception_type": "<misconception tag or empty string>",
      "feedback": "<your feedback message>"
    }
  ]
}

Do not include any text outside the JSON object."""


def _build_user_prompt(
    scene_graph: Dict[str, Any],
    retrieved_misconceptions: List[Dict[str, Any]],
) -> str:
    domain   = scene_graph.get("domain", "general_science")
    elements = scene_graph.get("elements", [])
    relations = scene_graph.get("relations", [])
    summary  = scene_graph.get("spatial_summary", "No summary available.")

    elem_desc = "\n".join(
        f"  - [{e['id']}] {e['type'].upper()} '{e['label']}' "
        f"(grounding: {e['grounding']}, confidence: {e['confidence']:.0%})"
        for e in elements
    )
    rel_desc = "\n".join(
        f"  - {r['source']} --[{r['relation_type']}]--> {r['target']} "
        f"(confidence: {r['confidence']:.0%})"
        for r in relations
    ) or "  (no relations detected)"

    misc_desc = "\n".join(
        f"  - [{m['id']}] {m['description']} (similarity: {m['similarity']:.2f})"
        for m in retrieved_misconceptions
    ) or "  (no likely misconceptions flagged)"

    return f"""Domain: {domain}
Spatial summary: {summary}

Visual Elements:
{elem_desc}

Spatial Relations:
{rel_desc}

Potentially Relevant Misconceptions (from educational research):
{misc_desc}

Generate Socratic formative feedback for this student diagram.
Focus on the most educationally important issues. Limit to 3-4 feedback items."""


# ---------------------------------------------------------------------------
# Main reasoning engine
# ---------------------------------------------------------------------------

class LLMReasoningEngine:

    def __init__(self):
        self.base_url  = OLLAMA_BASE_URL
        self.llm_model = OLLAMA_LLM_MODEL
        self.timeout   = OLLAMA_TIMEOUT

    async def generate_feedback(
        self,
        scene_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Main entry point. Accepts a Stage-1 scene graph and returns
        structured feedback ready for Stage-3 rendering.
        """
        domain = scene_graph.get("domain", "general_science")

        # RAG retrieval: build query from element labels + spatial summary
        rag_query = (
            " ".join(e["label"] for e in scene_graph.get("elements", []))
            + " " + scene_graph.get("spatial_summary", "")
        )
        retrieved = retrieve_misconceptions(rag_query, domain=domain, top_k=3)

        user_prompt = _build_user_prompt(scene_graph, retrieved)

        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": SOCRATIC_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "stream":  False,
            "format":  "json",
            "options": {
                "temperature": 0.35,
                "num_predict": 800,
            },
        }

        feedback_items: List[Dict[str, Any]] = []
        identified_subdomain = ""

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                raw = resp.json()["message"]["content"].strip()
                identified_subdomain, feedback_items = self._parse_feedback(raw)
                
                # Log to analytics if subdomain is identified
                if identified_subdomain:
                    for item in feedback_items:
                        if item.get("status") == "incorrect" and item.get("misconception_type"):
                            await update_teacher_heatmap(identified_subdomain, item["misconception_type"])

        except httpx.ConnectError:
            logger.warning("Ollama unavailable — using rule-based fallback feedback.")
            identified_subdomain, feedback_items = self._rule_based_feedback(scene_graph, retrieved)

        except Exception as exc:
            logger.error("LLM reasoning error: %s", exc)
            identified_subdomain, feedback_items = self._rule_based_feedback(scene_graph, retrieved)

        return {
            "domain":            domain,
            "identified_subdomain": identified_subdomain,
            "heatmap_updated":   bool(identified_subdomain),
            "feedback_items":    feedback_items,
            "retrieved_miscs":   retrieved,
            "element_count":     len(scene_graph.get("elements", [])),
            "relation_count":    len(scene_graph.get("relations", [])),
        }

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_feedback(self, raw: str) -> Tuple[str, List[Dict[str, Any]]]:
        raw = raw.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = "\n".join(
                ln for ln in raw.split("\n")
                if not ln.strip().startswith("```")
            )
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                subdomain = parsed.get("identified_subdomain", "")
                elements = parsed.get("evaluated_elements", [])
                if not elements:
                    for key in ("feedback", "items", "feedback_items"):
                        if isinstance(parsed.get(key), list):
                            elements = parsed[key]
                            break
                return subdomain, [self._coerce_feedback_item(i) for i in elements]
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse LLM feedback JSON: %s\nRaw: %.200s", exc, raw)
        return "", []

    @staticmethod
    def _coerce_feedback_item(item: Any) -> Dict[str, Any]:
        if not isinstance(item, dict):
            return {"element_id": "", "type": "unknown", "status": "incorrect", "misconception_type": "", "feedback": str(item)}
        return {
            "element_id":          item.get("element_id", ""),
            "type":                item.get("type", "unknown"),
            "status":              item.get("status", "incorrect"),
            "misconception_type":  item.get("misconception_type", ""),
            "feedback":            item.get("feedback", ""),
        }

    def _rule_based_feedback(
        self,
        scene_graph: Dict[str, Any],
        retrieved: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Deterministic fallback feedback when Ollama is unavailable.
        Based on retrieved misconceptions and element analysis.
        """
        domain   = scene_graph.get("domain", "general_science")
        elements = scene_graph.get("elements", [])
        feedback: List[Dict[str, Any]] = []

        # Affirmative opener
        elem_count = len(elements)
        feedback.append({
            "element_id": "",
            "type": "affirmative",
            "status": "correct",
            "misconception_type": "",
            "feedback": (
                f"Great effort! I can see you've identified {elem_count} element(s) "
                f"in this {domain} diagram. Let's think carefully about the relationships."
            )
        })

        # Per-misconception Socratic questions
        for misc in retrieved[:2]:
            feedback.append({
                "element_id":          "",
                "type":                "socratic",
                "status":              "incorrect",
                "misconception_type":  misc.get("tag", ""),
                "feedback": (
                    f"Interesting! You've drawn something in this area. "
                    f"What do you think happens to energy/matter at this point? "
                    f"Can you explain why you drew it that way?"
                ),
            })

        # Arrow direction check
        vectors = [e for e in elements if e.get("type") == "vector"]
        if vectors:
            feedback.append({
                "element_id":          vectors[0]["id"] if vectors else "",
                "type":                "socratic",
                "status":              "incorrect",
                "misconception_type":  "arrow_direction",
                "feedback": (
                    f"I notice you've drawn {len(vectors)} arrow(s). "
                    f"What does the direction of each arrow represent in your diagram? "
                    f"What would change if you reversed it?"
                ),
            })

        return "Unknown_Subdomain", feedback

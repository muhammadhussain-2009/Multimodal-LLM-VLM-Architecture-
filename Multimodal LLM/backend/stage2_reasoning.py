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
# Lightweight TF-IDF similarity matcher (zero-dependency RAG retrieval)
# ---------------------------------------------------------------------------

def _build_tfidf_index(
    docs: List[str],
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Build term-frequency vectors and IDF weights from a list of documents.
    Returns: (tf_vectors, idf_dict)
    """
    n = len(docs)
    tokenise = lambda text: re.findall(r'\b[a-z]{3,}\b', text.lower())

    tf_vectors: List[Dict[str, float]] = []
    df: Counter = Counter()

    for doc in docs:
        tokens = tokenise(doc)
        freq   = Counter(tokens)
        total  = len(tokens) or 1
        tf     = {t: c / total for t, c in freq.items()}
        tf_vectors.append(tf)
        df.update(set(freq.keys()))

    idf = {term: math.log((n + 1) / (cnt + 1)) + 1 for term, cnt in df.items()}
    return tf_vectors, idf


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    common = set(vec_a) & set(vec_b)
    dot    = sum(vec_a[t] * vec_b[t] for t in common)
    mag_a  = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b  = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    return dot / (mag_a * mag_b) if (mag_a and mag_b) else 0.0


# Pre-build index at import time (fast, in-memory)
_MISC_TEXTS = [m["description"] for m in MISCONCEPTION_LIBRARY]
_TF_VECTORS, _IDF = _build_tfidf_index(_MISC_TEXTS)


def retrieve_misconceptions(
    query: str,
    domain: Optional[str] = None,
    top_k: int = 3,
    threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    TF-IDF similarity search over misconception library.
    Optional domain filter reduces false positives cross-domain.
    """
    tokenise = lambda text: re.findall(r'\b[a-z]{3,}\b', text.lower())
    tokens   = tokenise(query)
    freq     = Counter(tokens)
    total    = len(tokens) or 1
    query_tf = {t: (c / total) * _IDF.get(t, 1.0) for t, c in freq.items()}

    scores = []
    for i, (tv, misc) in enumerate(zip(_TF_VECTORS, MISCONCEPTION_LIBRARY)):
        if domain and misc["domain"] != domain:
            continue
        tfidf_vec = {t: v * _IDF.get(t, 1.0) for t, v in tv.items()}
        sim = _cosine_similarity(query_tf, tfidf_vec)
        scores.append((sim, misc))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [
        {**m, "similarity": round(sim, 4)}
        for sim, m in scores[:top_k]
        if sim >= threshold
    ]


# ---------------------------------------------------------------------------
# Socratic prompt construction
# ---------------------------------------------------------------------------

SOCRATIC_SYSTEM_PROMPT = """You are Socratica, an expert STEM educator who gives
Socratic formative feedback to students. Your role is to:

1. NEVER directly correct a student — always respond with guiding questions.
2. Acknowledge what the student got right before addressing errors.
3. Address exactly ONE misconception per feedback item.
4. Use simple language appropriate for the student's grade level.
5. End each feedback item with a reflection question.

Return ONLY a JSON array of feedback objects:
[
  {
    "type": "socratic" | "affirmative" | "corrective",
    "text": "<your feedback message>",
    "misconception_class": "<misconception tag or empty string>",
    "target_element": "<element id this feedback addresses>",
    "confidence": 0.0 to 1.0
  }
]

Do not include any text outside the JSON array."""


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

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                raw = resp.json()["message"]["content"].strip()
                feedback_items = self._parse_feedback(raw)

        except httpx.ConnectError:
            logger.warning("Ollama unavailable — using rule-based fallback feedback.")
            feedback_items = self._rule_based_feedback(scene_graph, retrieved)

        except Exception as exc:
            logger.error("LLM reasoning error: %s", exc)
            feedback_items = self._rule_based_feedback(scene_graph, retrieved)

        return {
            "domain":            domain,
            "feedback_items":    feedback_items,
            "retrieved_miscs":   retrieved,
            "element_count":     len(scene_graph.get("elements", [])),
            "relation_count":    len(scene_graph.get("relations", [])),
        }

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_feedback(self, raw: str) -> List[Dict[str, Any]]:
        raw = raw.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = "\n".join(
                ln for ln in raw.split("\n")
                if not ln.strip().startswith("```")
            )
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                return [self._coerce_feedback_item(item) for item in arr]
            # Some models wrap in {"feedback": [...]}
            if isinstance(arr, dict):
                for key in ("feedback", "items", "feedback_items"):
                    if isinstance(arr.get(key), list):
                        return [self._coerce_feedback_item(i) for i in arr[key]]
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse LLM feedback JSON: %s\nRaw: %.200s", exc, raw)
        return []

    @staticmethod
    def _coerce_feedback_item(item: Any) -> Dict[str, Any]:
        if not isinstance(item, dict):
            return {"type": "socratic", "text": str(item), "misconception_class": "",
                    "target_element": "", "confidence": 0.7}
        return {
            "type":                item.get("type", "socratic"),
            "text":                item.get("text", ""),
            "misconception_class": item.get("misconception_class", ""),
            "target_element":      item.get("target_element", ""),
            "confidence":          float(item.get("confidence", 0.8)),
        }

    def _rule_based_feedback(
        self,
        scene_graph: Dict[str, Any],
        retrieved: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
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
            "type": "affirmative",
            "text": (
                f"Great effort! I can see you've identified {elem_count} element(s) "
                f"in this {domain} diagram. Let's think carefully about the relationships."
            ),
            "misconception_class": "",
            "target_element":      "",
            "confidence":          1.0,
        })

        # Per-misconception Socratic questions
        for misc in retrieved[:2]:
            feedback.append({
                "type": "socratic",
                "text": (
                    f"Interesting! You've drawn something in this area. "
                    f"What do you think happens to energy/matter at this point? "
                    f"Can you explain why you drew it that way?"
                ),
                "misconception_class": misc.get("tag", ""),
                "target_element":      "",
                "confidence":          misc.get("similarity", 0.5),
            })

        # Arrow direction check
        vectors = [e for e in elements if e.get("type") == "vector"]
        if vectors:
            feedback.append({
                "type": "socratic",
                "text": (
                    f"I notice you've drawn {len(vectors)} arrow(s). "
                    f"What does the direction of each arrow represent in your diagram? "
                    f"What would change if you reversed it?"
                ),
                "misconception_class": "arrow_direction",
                "target_element":      vectors[0]["id"] if vectors else "",
                "confidence":          0.85,
            })

        return feedback

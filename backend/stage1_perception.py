"""
backend/stage1_perception.py
=============================
Stage 1: VLM Perception Engine

Converts a pre-processed student STEM diagram into a structured scene graph
by prompting the local Ollama VLM endpoint.

Selected model: llava:13b (best open-source multimodal on Ollama for spatial
diagram understanding — outperforms 7b variants on AI2D benchmarks).

All NVIDIA NIM references removed. The system operates fully offline via Ollama.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("VLMPerception")

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_VLM_MODEL = os.getenv("OLLAMA_VLM_MODEL", "llava:13b")
OLLAMA_TIMEOUT   = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))

# ---------------------------------------------------------------------------
# System prompt for scene-graph extraction
# ---------------------------------------------------------------------------
PERCEPTION_SYSTEM_PROMPT = """You are an expert STEM diagram analyst.
Your task is to analyze the provided image and extract a structured scene graph.

Return ONLY a valid JSON object with the following schema:
{
  "domain": "<physics|biology|chemistry|general_science>",
  "elements": [
    {
      "id": "<unique_id>",
      "type": "<node|vector|label>",
      "label": "<text visible in diagram or inferred>",
      "grounding": "<relational_arrow|directed_arrow|label|geometric_shape|region_container|generic>",
      "bbox": [x_min, y_min, x_max, y_max],   // null if not determinable
      "confidence": 0.0 to 1.0
    }
  ],
  "relations": [
    {
      "source": "<element_id>",
      "target": "<element_id>",
      "relation_type": "<flows_to|causes|part_of|connected_to|labels|contains>",
      "confidence": 0.0 to 1.0
    }
  ],
  "spatial_summary": "<1-2 sentence description of the overall diagram layout>",
  "diagram_quality": "<clear|partially_clear|unclear>"
}

Do not include any text outside the JSON object. Do not use markdown code fences."""


PERCEPTION_USER_PROMPT = """Analyze this STEM diagram. Identify every visual element
(arrows, labels, shapes, containers, nodes) and all spatial relationships between them.
Pay special attention to directionality of arrows and what each label annotates."""


class VLMPerceptionEngine:
    """
    Calls Ollama's local llava:13b endpoint with a base64-encoded diagram image
    and returns a parsed scene-graph dict.
    """

    def __init__(self):
        self.base_url  = OLLAMA_BASE_URL
        self.vlm_model = OLLAMA_VLM_MODEL
        self.timeout   = OLLAMA_TIMEOUT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        image_bytes: bytes,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a pre-processed image and return a scene graph.

        Args:
            image_bytes: Raw PNG bytes (should already be 448×448 RGB from preprocessor).
            context:     Optional extra context string (e.g., "Grade 9 physics test").

        Returns:
            Parsed scene-graph dict. Falls back to structured mock on Ollama failure.
        """
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        user_msg = PERCEPTION_USER_PROMPT
        if context:
            user_msg += f"\n\nContext: {context}"

        payload = {
            "model": self.vlm_model,
            "messages": [
                {"role": "system",  "content": PERCEPTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_msg,
                    "images": [b64_image],
                },
            ],
            "stream":  False,
            "format":  "json",
            "options": {
                "temperature": 0.1,   # low temp for factual extraction
                "num_predict": 1024,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                raw_content = data["message"]["content"]
                scene_graph = self._parse_response(raw_content)
                scene_graph["_source"] = "ollama_vlm"
                return scene_graph

        except httpx.ConnectError:
            logger.warning(
                "Ollama is not running at %s. Returning structured fallback. "
                "Start Ollama with: ollama serve", self.base_url
            )
            return self._build_fallback_scene_graph("ollama_unavailable")

        except httpx.TimeoutException:
            logger.warning("Ollama request timed out after %ds.", self.timeout)
            return self._build_fallback_scene_graph("ollama_timeout")

        except Exception as exc:
            logger.error("VLM perception error: %s", exc)
            return self._build_fallback_scene_graph(f"error: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse the VLM's raw text output into a validated scene-graph dict."""
        # Strip any accidental markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                ln for ln in lines
                if not ln.strip().startswith("```")
            )

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse VLM JSON response: %s\nRaw: %.200s", exc, raw)
            obj = {}

        return self._validate_and_normalise(obj)

    def _validate_and_normalise(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """Fill in missing keys with safe defaults."""
        valid_domains = {"physics", "biology", "chemistry", "general_science"}
        domain = obj.get("domain", "general_science")
        if domain not in valid_domains:
            domain = "general_science"

        elements = []
        for i, el in enumerate(obj.get("elements", [])):
            elements.append({
                "id":         el.get("id", f"elem_{i}"),
                "type":       el.get("type", "node"),
                "label":      el.get("label", ""),
                "grounding":  el.get("grounding", "generic"),
                "bbox":       el.get("bbox"),
                "confidence": float(el.get("confidence", 0.9)),
            })

        relations = []
        for rel in obj.get("relations", []):
            relations.append({
                "source":        rel.get("source", ""),
                "target":        rel.get("target", ""),
                "relation_type": rel.get("relation_type", "connected_to"),
                "confidence":    float(rel.get("confidence", 0.8)),
            })

        return {
            "domain":          domain,
            "elements":        elements,
            "relations":       relations,
            "spatial_summary": obj.get("spatial_summary", ""),
            "diagram_quality": obj.get("diagram_quality", "clear"),
        }

    def _build_fallback_scene_graph(self, reason: str) -> Dict[str, Any]:
        """Return a minimal valid scene graph when Ollama is unavailable."""
        return {
            "domain":   "general_science",
            "elements": [
                {
                    "id": "elem_0", "type": "node", "label": "Diagram (Ollama offline)",
                    "grounding": "generic", "bbox": None, "confidence": 0.5,
                },
            ],
            "relations":       [],
            "spatial_summary": "VLM analysis unavailable — Ollama not running.",
            "diagram_quality": "unclear",
            "_source":         "mock_fallback",
            "_reason":         reason,
        }

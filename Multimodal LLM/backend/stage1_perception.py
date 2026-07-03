import os
import json
import base64
import asyncio
from typing import Dict, List, Any, Optional
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VLMPerception")

class VLMPerceptionEngine:
    """
    Stage 1: VLM Perception Engine.
    Handles spatial element segmentation, domain-specific symbol grounding, 
    and relational graph extraction from STEM diagrams.
    Supports local Ollama (Llava/MiniCPM-V) and cloud NVIDIA NIM (Llama-3.2-Vision).
    """
    def __init__(self, provider: str = "mock", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        
    def _image_to_base64(self, image_path: str) -> str:
        """Helper to read image file and convert to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    async def extract_scene_graph(self, image_path: str, domain: str = "physics") -> Dict[str, Any]:
        """
        Extract scene graph from the diagram.
        
        Args:
            image_path: Path to diagram image file.
            domain: The STEM domain ('physics', 'biology', 'chemistry').
            
        Returns:
            A structured Scene Graph JSON dictionary.
        """
        logger.info(f"Processing diagram image with {self.provider} in domain {domain}...")
        
        if self.provider == "mock":
            await asyncio.sleep(1.0) # Simulate latency
            return self._get_mock_scene_graph(domain)
            
        try:
            base64_image = self._image_to_base64(image_path)
            
            prompt = self._get_system_prompt(domain)
            
            if self.provider == "ollama":
                return await self._call_ollama(base64_image, prompt)
            elif self.provider == "nvidia":
                return await self._call_nvidia_nim(base64_image, prompt)
            else:
                raise ValueError(f"Unknown VLM provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"VLM inference failed: {str(e)}. Falling back to mock data.")
            return self._get_mock_scene_graph(domain)

    def _get_system_prompt(self, domain: str) -> str:
        return f"""
        You are a highly precise VLM specialized in STEM diagram parsing.
        Analyze this diagram in the domain of: {domain}.
        
        Task:
        1. Perform spatial element segmentation. Identify all elements (labels, arrows, shapes) with bounding boxes normalized from 0 to 1000 [ymin, xmin, ymax, xmax].
        2. Perform symbol grounding:
           - If domain is 'physics': Distinguish force vectors (e.g. Gravity, Friction, Normal) from direction arrows or structural components.
           - If domain is 'biology': Ground arrows as flows/pathways or cycle stages.
           - If domain is 'chemistry': Ground lines as covalent bonds, reaction pathways, or equilibrium arrows.
        3. Relational graph extraction: Link elements together (e.g. which force vector acts on which object node, which chemical bond connects which atom).
        4. Give a confidence score (0.0 to 1.0) for every element and relation.

        Return ONLY a raw JSON block matching this structure:
        {{
          "elements": [
            {{
              "id": "e1",
              "type": "vector" | "label" | "node",
              "label": "name of element",
              "bbox": [ymin, xmin, ymax, xmax],
              "confidence": 0.95,
              "grounding": "physics_force_vector" | "biology_cycle_arrow" | "chemical_bond" | "generic"
            }}
          ],
          "relations": [
            {{
              "source": "e1",
              "target": "e2",
              "relation_type": "acts_on" | "points_to" | "connected_to" | "transforms_into",
              "confidence": 0.88
            }}
          ]
        }}
        """

    async def _call_ollama(self, base64_image: str, prompt: str) -> Dict[str, Any]:
        """Call local Ollama VLM instance (e.g., llava or minicpm-v)."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": "llava",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64_image]
                }
            ],
            "stream": False,
            "format": "json"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            content = result["message"]["content"]
            return json.loads(content)

    async def _call_nvidia_nim(self, base64_image: str, prompt: str) -> Dict[str, Any]:
        """Call cloud NVIDIA NIM Llama 3.2 Vision or similar VLM."""
        if not self.api_key:
            raise ValueError("NVIDIA NIM API key is missing. Set NVIDIA_API_KEY environment variable.")
            
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "nvidia/llama-3.2-11b-vision-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)

    def _get_mock_scene_graph(self, domain: str) -> Dict[str, Any]:
        """Fallback mock graphs representing typical student mistakes to showcase the pipeline."""
        if domain == "physics":
            return {
                "elements": [
                    {"id": "node_block", "type": "node", "label": "Block on Ramp", "bbox": [400, 450, 550, 600], "confidence": 0.98, "grounding": "object"},
                    {"id": "node_ramp", "type": "node", "label": "Incline Ramp (30 deg)", "bbox": [500, 200, 800, 800], "confidence": 0.99, "grounding": "object"},
                    {"id": "vec_gravity", "type": "vector", "label": "Gravity Vector (Fg)", "bbox": [480, 520, 700, 520], "confidence": 0.94, "grounding": "physics_force_vector"},
                    {"id": "vec_friction", "type": "vector", "label": "Friction (Ff)", "bbox": [480, 520, 550, 320], "confidence": 0.89, "grounding": "physics_force_vector"},
                    {"id": "lbl_theta", "type": "label", "label": "Angle theta", "bbox": [750, 720, 780, 770], "confidence": 0.92, "grounding": "label"}
                ],
                "relations": [
                    {"source": "vec_gravity", "target": "node_block", "relation_type": "acts_on", "confidence": 0.95},
                    {"source": "vec_friction", "target": "node_block", "relation_type": "acts_on", "confidence": 0.90},
                    {"source": "node_block", "target": "node_ramp", "relation_type": "resting_on", "confidence": 0.97}
                ]
            }
        elif domain == "biology":
            return {
                "elements": [
                    {"id": "cell_wall", "type": "node", "label": "Cell Wall", "bbox": [100, 100, 900, 900], "confidence": 0.99, "grounding": "cellular_boundary"},
                    {"id": "nucleus", "type": "node", "label": "Nucleus", "bbox": [450, 450, 600, 600], "confidence": 0.97, "grounding": "organelle"},
                    {"id": "chloroplast_1", "type": "node", "label": "Mitochondrion", "bbox": [200, 300, 300, 450], "confidence": 0.85, "grounding": "organelle"},
                    {"id": "vacuole", "type": "node", "label": "Large Central Vacuole", "bbox": [300, 620, 800, 850], "confidence": 0.96, "grounding": "organelle"}
                ],
                "relations": [
                    {"source": "nucleus", "target": "cell_wall", "relation_type": "inside_of", "confidence": 0.99},
                    {"source": "chloroplast_1", "target": "cell_wall", "relation_type": "inside_of", "confidence": 0.99},
                    {"source": "vacuole", "target": "cell_wall", "relation_type": "inside_of", "confidence": 0.99}
                ]
            }
        else:
            return {
                "elements": [
                    {"id": "atom_o", "type": "node", "label": "O", "bbox": [450, 450, 550, 550], "confidence": 0.98, "grounding": "chemical_element"},
                    {"id": "atom_h1", "type": "node", "label": "H", "bbox": [450, 200, 550, 300], "confidence": 0.95, "grounding": "chemical_element"},
                    {"id": "atom_h2", "type": "node", "label": "H", "bbox": [450, 700, 550, 800], "confidence": 0.95, "grounding": "chemical_element"},
                    {"id": "bond_1", "type": "vector", "label": "Single Bond", "bbox": [500, 300, 500, 450], "confidence": 0.92, "grounding": "chemical_bond"},
                    {"id": "bond_2", "type": "vector", "label": "Single Bond", "bbox": [500, 550, 500, 700], "confidence": 0.92, "grounding": "chemical_bond"}
                ],
                "relations": [
                    {"source": "bond_1", "target": "atom_o", "relation_type": "connected_to", "confidence": 0.95},
                    {"source": "bond_1", "target": "atom_h1", "relation_type": "connected_to", "confidence": 0.95},
                    {"source": "bond_2", "target": "atom_o", "relation_type": "connected_to", "confidence": 0.95},
                    {"source": "bond_2", "target": "atom_h2", "relation_type": "connected_to", "confidence": 0.95}
                ]
            }

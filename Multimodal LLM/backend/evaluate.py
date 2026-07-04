import time
from typing import Dict, List, Any, Set
import math

class PipelineEvaluator:
    """
    Implements validation metrics and benchmark statistics for evaluating
    pedagogical effectiveness, perception engine accuracy, and inference latency.
    """
    
    @staticmethod
    def calculate_normalized_learning_gain(pre_test: float, post_test: float) -> float:
        """
        Normalized Learning Gain g = (Post - Pre) / (FullScore - Pre).
        Represents what proportion of the maximum potential gain was achieved.
        """
        max_score = 100.0
        if pre_test >= max_score:
            return 0.0 if post_test >= max_score else (post_test - pre_test) / (max_score - pre_test)
        
        if post_test < pre_test:
            # Negative learning gain (score decreased)
            return (post_test - pre_test) / pre_test
            
        return (post_test - pre_test) / (max_score - pre_test)

    @staticmethod
    def calculate_classification_f1(tp: int, fp: int, fn: int) -> float:
        """Computes F1 Score for misconception classification accuracy."""
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if (precision + recall) == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def calculate_graph_edit_distance(graph_a: Dict[str, Any], graph_b: Dict[str, Any]) -> float:
        """
        Calculates Graph Edit Distance (GED) between two scene graphs.
        Uses element ids, types, labels, and relations to compute edit operations:
        - Insertion, deletion, substitution of nodes and edges.
        """
        # Node mapping
        nodes_a = {e["id"]: (e.get("type"), e.get("label")) for e in graph_a.get("elements", [])}
        nodes_b = {e["id"]: (e.get("type"), e.get("label")) for e in graph_b.get("elements", [])}
        
        # Count node substitutions, deletions, insertions
        matched_nodes = 0
        for nid, attributes in nodes_a.items():
            if nid in nodes_b:
                if attributes == nodes_b[nid]:
                    matched_nodes += 1
                    
        node_deletes = len(nodes_a) - matched_nodes
        node_inserts = len(nodes_b) - matched_nodes
        
        # Edge mapping
        edges_a = {(r["source"], r["target"], r.get("relation_type")) for r in graph_a.get("relations", [])}
        edges_b = {(r["source"], r["target"], r.get("relation_type")) for r in graph_b.get("relations", [])}
        
        matched_edges = len(edges_a.intersection(edges_b))
        edge_deletes = len(edges_a) - matched_edges
        edge_inserts = len(edges_b) - matched_edges
        
        # Total GED is sum of node edits and edge edits
        total_ged = node_deletes + node_inserts + edge_deletes + edge_inserts
        return float(total_ged)

    @staticmethod
    def llm_as_a_judge_socratic_score(socratic_feedback: str, correct_concept: str) -> Dict[str, Any]:
        """
        Evaluates feedback quality along four pedagogic criteria on a 1-5 scale:
        1. Socratic Nature (asks questions rather than telling answers)
        2. Correctness (identifies the actual conceptual mistake correctly)
        3. Scaffolding (provides intermediate steps/hints)
        4. Tone (encouraging and age-appropriate)
        """
        # Rule-based judge simulation helper
        socratic_keywords = ["why", "how", "what", "think", "notice", "remember", "?"]
        tells_answer = any(phrase in socratic_feedback.lower() for phrase in [
            "the correct answer is", "you should draw a", "here is the answer", "draw it like this"
        ])
        
        # Criteria scoring
        socratic_score = 5
        if "?" not in socratic_feedback:
            socratic_score -= 2
        if tells_answer:
            socratic_score -= 3
            
        correctness = 5 if correct_concept.lower() in socratic_feedback.lower() else 4
        scaffolding = 5 if len(socratic_feedback.split(".")) > 3 else 3
        tone = 5 if any(w in socratic_feedback.lower() for w in ["great", "good job", "look closely", "first"]) else 4
        
        avg_score = (socratic_score + correctness + scaffolding + tone) / 4.0
        
        return {
            "socratic_dialogue_metric": socratic_score,
            "concept_match_metric": correctness,
            "cognitive_scaffolding_metric": scaffolding,
            "encouragement_tone_metric": tone,
            "composite_pedagogical_index": avg_score
        }

    @staticmethod
    def hardware_latency_matrix(tokens_generated: int) -> Dict[str, float]:
        """
        Returns estimated inference execution speeds (ms/token) mapped
        across deployment target tiers to demonstrate local vs cloud compromises.
        """
        # Normalized hardware speed measurements (in ms per token)
        hardware_tiers = {
            "Raspberry_Pi_4_Edge_Local": 620.0,
            "School_Server_CPU_Local": 110.0,
            "Teacher_Laptop_Intel_Core_i7": 85.0,
            "Nvidia_Jetson_Orin_Nano": 45.0,
            "Nvidia_NIM_Cloud_Llama3": 12.0
        }
        
        result = {}
        for tier, ms_per_token in hardware_tiers.items():
            result[tier] = float(tokens_generated * ms_per_token)
            
        return result

import json
from typing import List, Dict, Any

class FeedbackRenderer:
    """
    Stage 3: Adaptive Feedback Rendering Engine.
    Handles coordinate mapping, heatmap aggregation, accessibility summaries, 
    and longitudinal tracking calculations.
    """
    @staticmethod
    def map_bboxes_to_viewport(
        elements: List[Dict[str, Any]], 
        width: int, 
        height: int
    ) -> List[Dict[str, Any]]:
        """
        Converts normalized coordinates [ymin, xmin, ymax, xmax] (0 to 1000 scale)
        to absolute pixel values for the frontend canvas rendering.
        """
        mapped = []
        for elem in elements:
            bbox = elem.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
                
            ymin, xmin, ymax, xmax = bbox
            
            # Map normalized 1000x1000 coordinates to viewport width/height
            px_left = int((xmin / 1000.0) * width)
            px_top = int((ymin / 1000.0) * height)
            px_width = int(((xmax - xmin) / 1000.0) * width)
            px_height = int(((ymax - ymin) / 1000.0) * height)
            
            mapped.append({
                "id": elem["id"],
                "type": elem["type"],
                "label": elem["label"],
                "grounding": elem.get("grounding", "generic"),
                "confidence": elem.get("confidence", 1.0),
                "pixel_coords": {
                    "left": px_left,
                    "top": px_top,
                    "width": px_width,
                    "height": px_height
                }
            })
        return mapped

    @staticmethod
    def aggregate_class_heatmaps(submissions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates class submissions to generate misconception distribution.
        
        Returns:
            {
                "misconception_frequencies": {"PHYS-M1": 14, "PHYS-M2": 8},
                "total_students_assessed": 28,
                "attention_needed": ["PHYS-M1"] # Any misconception with > 40% prevalence
            }
        """
        frequencies = {}
        student_set = set()
        
        for sub in submissions:
            student_id = sub.get("student_id", "anonymous")
            student_set.add(student_id)
            
            misconceptions = sub.get("misconceptions", [])
            for m in misconceptions:
                m_id = m.get("id")
                if m_id:
                    frequencies[m_id] = frequencies.get(m_id, 0) + 1
                    
        total_students = len(student_set) if student_set else 1
        attention_needed = []
        for m_id, count in frequencies.items():
            prevalence = count / total_students
            if prevalence >= 0.40:
                attention_needed.append(m_id)
                
        return {
            "misconception_frequencies": frequencies,
            "total_students_assessed": len(student_set),
            "attention_needed": attention_needed
        }

    @staticmethod
    def calculate_persistence_metrics(history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates longitudinal persistence of misconceptions across chronological attempts.
        
        Args:
            history: List of submission events sorted by timestamp for a single student.
                     Example record: {"timestamp": "2026-07-01", "misconceptions": ["PHYS-M1", "PHYS-M2"]}
            
        Returns:
            A profile containing persistence count per misconception.
        """
        persistence = {}
        first_seen = {}
        last_seen = {}
        resolved = {}
        
        for idx, attempt in enumerate(history):
            misconceptions = attempt.get("misconceptions", [])
            timestamp = attempt.get("timestamp", str(idx))
            
            for m_id in misconceptions:
                if m_id not in first_seen:
                    first_seen[m_id] = idx
                last_seen[m_id] = idx
                # Reset resolved status if it appeared again
                resolved[m_id] = False
                
            # Check resolved misconceptions (present previously but not in this attempt)
            for m_id in list(first_seen.keys()):
                if m_id not in misconceptions and not resolved[m_id] and last_seen[m_id] < idx:
                    resolved[m_id] = True
                    
        for m_id, start_idx in first_seen.items():
            end_idx = last_seen[m_id]
            attempts_active = (end_idx - start_idx) + 1
            persistence[m_id] = {
                "attempts_active": attempts_active,
                "currently_resolved": resolved.get(m_id, False),
                "trend": "resolved" if resolved.get(m_id, False) else ("persistent" if attempts_active > 2 else "new")
            }
            
        return persistence

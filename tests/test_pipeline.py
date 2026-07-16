import pytest
import unittest.mock
import httpx
from backend.image_preprocessor import preprocess_image
from backend.stage1_perception import VLMPerceptionEngine
from backend.stage2_reasoning import LLMReasoningEngine, retrieve_misconceptions

class TestImagePreprocessor:
    def test_preprocess_valid_png(self, sample_image_bytes):
        # Even though validate_and_normalise expects a PIL Image conceptually,
        # it actually takes raw bytes in our implementation and returns RGB 448x448 bytes.
        result = preprocess_image(sample_image_bytes)
        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_validate_image_rejects_corrupt(self, corrupt_image_bytes):
        with pytest.raises(ValueError, match="Unable to decode the uploaded file as an image"):
            preprocess_image(corrupt_image_bytes)

class TestMisconceptionRetrieval:
    def test_retrieve_physics_misconceptions(self):
        results = retrieve_misconceptions("force vector gravity", domain="physics")
        assert isinstance(results, list)
        
    def test_retrieve_empty_query(self):
        results = retrieve_misconceptions("")
        assert len(results) == 0

@pytest.mark.asyncio
class TestPerceptionEngine:
    @unittest.mock.patch('httpx.AsyncClient.post')
    async def test_analyze_ollama_unavailable(self, mock_post, sample_image_bytes):
        mock_post.side_effect = httpx.ConnectError("Connection refused")
        
        engine = VLMPerceptionEngine()
        result = await engine.analyze(
            image_bytes=sample_image_bytes, 
            context="physics"
        )
        
        assert "_source" in result
        assert result["_source"] == "mock_fallback"
        assert "domain" in result
        assert "elements" in result
        
    @unittest.mock.patch('httpx.AsyncClient.post')
    async def test_analyze_timeout(self, mock_post, sample_image_bytes):
        mock_post.side_effect = httpx.TimeoutException("Read timeout")
        
        engine = VLMPerceptionEngine()
        result = await engine.analyze(
            image_bytes=sample_image_bytes, 
            context="physics"
        )
        
        assert "_source" in result
        assert result["_source"] == "mock_fallback"

@pytest.mark.asyncio
class TestReasoningEngine:
    @unittest.mock.patch('httpx.AsyncClient.post')
    @unittest.mock.patch('backend.stage2_reasoning.update_teacher_heatmap')
    async def test_generate_feedback_success(self, mock_heatmap, mock_post, sample_scene_graph):
        mock_response = unittest.mock.Mock()
        mock_response.json.return_value = {
            "message": {
                "content": '''
                {
                    "identified_subdomain": "Physics_Kinematics",
                    "evaluated_elements": [
                        {"element_id": "e1", "type": "arrow", "status": "correct", "feedback": "Good job.", "misconception_type": ""},
                        {"element_id": "e2", "type": "box", "status": "incorrect", "feedback": "Think again.", "misconception_type": "gravity_misconception"}
                    ]
                }
                '''
            }
        }
        mock_post.return_value = mock_response
        
        engine = LLMReasoningEngine()
        result = await engine.generate_feedback(sample_scene_graph)
        
        assert "identified_subdomain" in result
        assert result["identified_subdomain"] == "Physics_Kinematics"
        assert result["heatmap_updated"] is True
        assert "feedback_items" in result
        assert len(result["feedback_items"]) == 2
        
        mock_heatmap.assert_called_once_with("Physics_Kinematics", "gravity_misconception")

    @unittest.mock.patch('httpx.AsyncClient.post')
    async def test_generate_feedback_fallback(self, mock_post, sample_scene_graph):
        mock_post.side_effect = httpx.ConnectError("Connection refused")
        
        engine = LLMReasoningEngine()
        result = await engine.generate_feedback(sample_scene_graph)
        
        assert "identified_subdomain" in result
        assert result["identified_subdomain"] == "Unknown_Subdomain"
        assert "feedback_items" in result
        assert len(result["feedback_items"]) >= 1
        
        statuses = [item.get("status") for item in result["feedback_items"]]
        assert "correct" in statuses or "incorrect" in statuses

import pytest
import unittest.mock
import httpx
from backend.image_preprocessor import preprocess_image
from backend.stage1_perception import analyze_diagram_with_vlm
from backend.stage2_reasoning import generate_socratic_feedback, retrieve_misconceptions

class TestImagePreprocessor:
    def test_preprocess_valid_png(self, sample_image_bytes):
        # Even though validate_and_normalise expects a PIL Image conceptually,
        # it actually takes raw bytes in our implementation and returns RGB 448x448 bytes.
        result = preprocess_image(sample_image_bytes)
        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_validate_image_rejects_corrupt(self, corrupt_image_bytes):
        with pytest.raises(ValueError, match="Invalid or corrupt image"):
            preprocess_image(corrupt_image_bytes)

class TestMisconceptionRetrieval:
    def test_retrieve_physics_misconceptions(self):
        results = retrieve_misconceptions("force vector gravity", domain_filter="physics")
        assert isinstance(results, list)
        
    def test_retrieve_empty_query(self):
        results = retrieve_misconceptions("")
        assert len(results) == 0

@pytest.mark.asyncio
class TestPerceptionEngine:
    @unittest.mock.patch('httpx.AsyncClient.post')
    async def test_analyze_ollama_unavailable(self, mock_post, sample_image_bytes):
        mock_post.side_effect = httpx.ConnectError("Connection refused")
        
        result = await analyze_diagram_with_vlm(
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
        
        result = await analyze_diagram_with_vlm(
            image_bytes=sample_image_bytes, 
            context="physics"
        )
        
        assert "_source" in result
        assert result["_source"] == "mock_fallback"

@pytest.mark.asyncio
class TestReasoningEngine:
    @unittest.mock.patch('httpx.AsyncClient.post')
    async def test_generate_feedback_fallback(self, mock_post, sample_scene_graph):
        mock_post.side_effect = httpx.ConnectError("Connection refused")
        
        result = await generate_socratic_feedback(sample_scene_graph)
        
        assert "feedback_items" in result
        assert len(result["feedback_items"]) >= 2
        # Verify it has at least one affirmative and one socratic fallback
        types = [item.get("type") for item in result["feedback_items"]]
        assert "affirmative" in types
        assert "socratic" in types

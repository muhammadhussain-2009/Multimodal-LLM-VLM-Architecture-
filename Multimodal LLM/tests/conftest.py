import struct
import zlib
import pytest
from httpx import AsyncClient

# Mock the environment before importing app
import os
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["SESSION_SECRET_KEY"] = "test-secret-key-12345678901234567890123456789012"

from backend.main import app

@pytest.fixture
async def test_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def sample_image_bytes():
    """Generates a valid 10x10 red PNG programmatically."""
    width, height = 10, 10
    
    # 8-byte PNG signature
    png_signature = b'\x89PNG\r\n\x1a\n'
    
    # IHDR chunk
    ihdr_data = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
    ihdr_chunk = struct.pack("!I", len(ihdr_data)) + b'IHDR' + ihdr_data + struct.pack("!I", ihdr_crc)
    
    # IDAT chunk (pixel data: filter byte + RGB for each pixel)
    # 10 rows of 10 red pixels (255, 0, 0). 
    # Each row is 1 byte filter (0) + 10 * 3 bytes (R, G, B) = 31 bytes per row
    raw_data = b''
    for _ in range(height):
        raw_data += b'\x00' + (b'\xff\x00\x00' * width)
        
    compressed_data = zlib.compress(raw_data)
    idat_crc = zlib.crc32(b'IDAT' + compressed_data) & 0xffffffff
    idat_chunk = struct.pack("!I", len(compressed_data)) + b'IDAT' + compressed_data + struct.pack("!I", idat_crc)
    
    # IEND chunk
    iend_crc = zlib.crc32(b'IEND') & 0xffffffff
    iend_chunk = struct.pack("!I", 0) + b'IEND' + struct.pack("!I", iend_crc)
    
    return png_signature + ihdr_chunk + idat_chunk + iend_chunk

@pytest.fixture
def corrupt_image_bytes():
    return os.urandom(1024)

@pytest.fixture
def large_image_bytes():
    return b'\x00' * (6 * 1024 * 1024)  # 6 MB

@pytest.fixture
def sample_scene_graph():
    return {
        "domain": "physics",
        "elements": [
            {
                "id": "elem_0",
                "type": "node",
                "label": "Block",
                "grounding": "geometric_shape",
                "bbox": [0.1, 0.1, 0.5, 0.5],
                "confidence": 0.95
            },
            {
                "id": "elem_1",
                "type": "vector",
                "label": "Force Arrow",
                "grounding": "directed_arrow",
                "bbox": [0.5, 0.5, 0.8, 0.8],
                "confidence": 0.9
            }
        ],
        "relations": [
            {
                "source": "elem_1",
                "target": "elem_0",
                "relation_type": "acts_on",
                "confidence": 0.85
            }
        ],
        "spatial_summary": "A block with a force arrow acting on it.",
        "diagram_quality": "clear"
    }

@pytest.fixture
def sample_feedback_result():
    return {
        "domain": "physics",
        "feedback_items": [
            {
                "type": "affirmative",
                "text": "Good job identifying the block.",
                "misconception_class": "",
                "target_element": "elem_0",
                "confidence": 1.0
            }
        ],
        "retrieved_miscs": [],
        "element_count": 2,
        "relation_count": 1
    }

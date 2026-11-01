import pytest
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from backend.services.preprocess_service import preprocess_service
from backend.services.reconstruction_service import reconstruction_service
from backend.services.mesh_service import mesh_service
from backend.services.cloud_ai_service import cloud_ai_service
from backend.core.config import UPLOADS_DIR, PROCESSED_DIR, MODELS_DIR

def create_shape_image(filename: str, shape: str = "vase"):
    path = UPLOADS_DIR / filename
    img = Image.new("RGBA", (300, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    if shape == "vase":
        # Draw symmetrical vase shape
        draw.polygon([(100, 50), (200, 50), (180, 120), (230, 220), (70, 220), (120, 120)], fill=(20, 20, 30, 255))
    elif shape == "cup":
        # Draw mug with handle
        draw.rectangle([80, 80, 200, 240], fill=(30, 40, 50, 255))
        draw.arc([170, 110, 250, 210], start=270, end=90, fill=(30, 40, 50, 255), width=15)
    img.save(path)
    return path

def test_preprocess_contour_and_edges():
    path = create_shape_image("test_vase_input.png", "vase")
    result = preprocess_service.process_sketch(path, threshold_val=220, remove_bg=True)
    assert result["contours_detected"] >= 1
    assert result["coverage_ratio"] > 0.05
    assert Path(result["processed_path"]).exists()
    assert Path(result["preview_path"]).exists()

def test_reconstruction_manifold_geometry():
    path = create_shape_image("test_cup_input.png", "cup")
    prep = preprocess_service.process_sketch(path)
    
    model_id = "test_unit_model_cup"
    res = reconstruction_service.generate_3d_mesh(
        processed_mask_path=prep["processed_path"],
        original_sketch_path=path,
        model_id=model_id,
        resolution="fast",
        smoothness=2
    )
    
    assert res["glb_path"].exists()
    assert res["obj_path"].exists()
    metrics = res["metrics"]
    assert metrics["vertex_count"] > 200
    assert metrics["face_count"] > 200
    assert metrics["is_watertight"] is True
    assert len(metrics["dimensions"]) == 3
    assert all(d > 0 for d in metrics["dimensions"])

def test_insufficient_sketch_coverage():
    # Blank image
    blank_path = UPLOADS_DIR / "blank_sketch.png"
    Image.new("RGBA", (100, 100), (255, 255, 255, 255)).save(blank_path)
    prep = preprocess_service.process_sketch(blank_path)
    
    with pytest.raises(ValueError, match="insufficient coverage"):
        reconstruction_service.generate_3d_mesh(
            processed_mask_path=prep["processed_path"],
            original_sketch_path=blank_path,
            model_id="blank_test"
        )

def test_cloud_ai_unconfigured_rejection():
    # When TRIPO_API_KEY is not set or empty, generate_with_tripo must reject cleanly without falsifying output
    import asyncio
    test_img = UPLOADS_DIR / "test_vase_input.png"
    if not test_img.exists():
        test_img = create_shape_image("test_vase_input.png", "vase")
    
    # If no key set, must raise ValueError
    if not cloud_ai_service.is_tripo_configured():
        with pytest.raises(ValueError, match="TRIPO_API_KEY is not configured"):
            asyncio.run(cloud_ai_service.generate_with_tripo(test_img, "test_cloud_model"))

def test_groq_vision_sketch_analysis():
    import asyncio
    from backend.services.groq_service import groq_service
    path = create_shape_image("test_cup_groq.png", "cup")
    result = asyncio.run(groq_service.analyze_sketch(path))
    assert "object_type" in result
    assert "description" in result
    assert "suggested_thickness" in result
    assert "suggested_smoothness" in result
    assert "visual_features" in result
    assert isinstance(result["visual_features"], list)
    if result.get("available"):
        model_used = result.get("model_used", "").lower()
        assert "qwen" in model_used or "computer vision" in model_used
    else:
        # Graceful fallback when rate-limited or external API busy
        assert "error" in result or "reason" in result

def test_groq_service_response_parsing(monkeypatch):
    import asyncio
    from backend.services.groq_service import groq_service
    import httpx

    groq_service.clear_cache()

    # Mock successful Groq response to verify parser logic deterministically
    mock_payload = {
        "choices": [
            {
                "message": {
                    "content": '```json\n{"object_type": "mug", "description": "Coffee mug with side handle", "suggested_thickness": 1.4, "suggested_smoothness": 4, "visual_features": ["Cylindrical body", "Side handle", "Flat bottom"]}\n```'
                }
            }
        ]
    }

    class MockResponse:
        status_code = 200
        def json(self):
            return mock_payload

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def post(self, *args, **kwargs):
            return MockResponse()

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    path = create_shape_image("test_mug_mock.png", "cup")
    result = asyncio.run(groq_service.analyze_sketch(path))

    assert result["available"] is True
    assert result["object_type"] == "mug"
    assert result["suggested_thickness"] == 1.4
    assert result["suggested_smoothness"] == 4
    assert len(result["visual_features"]) == 3
    assert "Groq LPU Vision" in result["model_used"]



import pytest
import io
import time
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from backend.main import app
from backend.core.config import settings

client = TestClient(app)

def create_sample_image_bytes(format="PNG", color=(0, 0, 0), bg=(255, 255, 255)):
    img = Image.new("RGBA", (200, 200), (*bg, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 40, 160, 160], fill=(*color, 255))
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "hardware" in data
    assert "available_engines" in data
    assert "local_manifold" in data["available_engines"]

def test_image_upload_success():
    img_bytes = create_sample_image_bytes()
    response = client.post(
        "/api/sketch/upload",
        files={"file": ("sketch.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "image_id" in data
    assert data["filename"].endswith(".png")
    assert data["width"] == 200
    assert data["height"] == 200

def test_image_upload_invalid_type():
    response = client.post(
        "/api/sketch/upload",
        files={"file": ("malicious.exe", b"fake binary executable", "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "Invalid file extension" in response.json()["detail"]

def test_image_upload_corrupted():
    response = client.post(
        "/api/sketch/upload",
        files={"file": ("broken.png", b"not a real png header", "image/png")}
    )
    assert response.status_code == 400
    assert "corrupted or not a valid image" in response.json()["detail"]

def test_image_upload_oversized():
    huge_data = b"0" * (settings.MAX_UPLOAD_SIZE_BYTES + 1024)
    response = client.post(
        "/api/sketch/upload",
        files={"file": ("huge.png", huge_data, "image/png")}
    )
    assert response.status_code == 413

def test_base64_upload_success():
    import base64
    img_bytes = create_sample_image_bytes()
    b64_str = "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")
    response = client.post(
        "/api/sketch/upload/base64",
        json={"image_base64": b64_str, "filename": "canvas.png"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "image_id" in data
    assert data["width"] == 200

def test_preprocess_endpoint():
    # 1. Upload
    img_bytes = create_sample_image_bytes()
    upload_res = client.post(
        "/api/sketch/upload",
        files={"file": ("circle.png", img_bytes, "image/png")}
    )
    image_id = upload_res.json()["image_id"]

    # 2. Preprocess
    proc_res = client.post(
        "/api/image/preprocess",
        json={
            "image_id": image_id,
            "threshold": 220,
            "remove_background": True,
            "invert": False,
            "edge_detection": True
        }
    )
    assert proc_res.status_code == 200
    data = proc_res.json()
    assert "processed_id" in data
    assert data["contours_detected"] >= 1
    assert data["coverage_ratio"] > 0

def test_preprocess_nonexistent_image():
    response = client.post(
        "/api/image/preprocess",
        json={"image_id": "non_existent_12345"}
    )
    assert response.status_code == 404

def test_generate_job_lifecycle_and_model_export():
    # 1. Upload sketch
    img_bytes = create_sample_image_bytes()
    upload_res = client.post(
        "/api/sketch/upload",
        files={"file": ("sphere_sketch.png", img_bytes, "image/png")}
    )
    image_id = upload_res.json()["image_id"]

    # 2. Trigger generation
    gen_res = client.post(
        "/api/generate/3d",
        json={
            "image_id": image_id,
            "prompt": "Smooth ceramic sphere",
            "engine": "local_manifold",
            "resolution": "fast",
            "smoothness": 2
        }
    )
    assert gen_res.status_code == 200
    gen_data = gen_res.json()
    job_id = gen_data["job_id"]
    model_id = gen_data["model_id"]
    assert job_id.startswith("job_")
    assert model_id.startswith("model_")

    # 3. Poll for completion (should take 2-4 seconds)
    max_wait = 20
    start = time.time()
    completed = False
    while time.time() - start < max_wait:
        status_res = client.get(f"/api/generate/status/{job_id}")
        assert status_res.status_code == 200
        status_data = status_res.json()
        if status_data["status"] == "completed":
            completed = True
            break
        elif status_data["status"] == "failed":
            pytest.fail(f"Job failed with error: {status_data.get('error')}")
        time.sleep(0.5)

    assert completed, "Generation job did not complete within timeout"

    # 4. Check model metadata
    model_res = client.get(f"/api/model/{model_id}")
    assert model_res.status_code == 200
    model_meta = model_res.json()
    assert model_meta["model_id"] == model_id
    assert model_meta["geometry"]["vertex_count"] > 100
    assert model_meta["geometry"]["face_count"] > 100
    assert len(model_meta["geometry"]["dimensions"]) == 3

    # 5. Check GLB download
    glb_res = client.get(f"/api/model/{model_id}/download?format=glb")
    assert glb_res.status_code == 200
    assert len(glb_res.content) > 1000
    assert glb_res.headers["content-type"] == "model/gltf-binary"

    # 6. Check OBJ download
    obj_res = client.get(f"/api/model/{model_id}/download?format=obj")
    assert obj_res.status_code == 200
    assert len(obj_res.content) > 1000
    assert "v " in obj_res.text  # standard OBJ vertex format

    # 7. Check History
    hist_res = client.get("/api/history")
    assert hist_res.status_code == 200
    hist_items = hist_res.json()
    assert any(h["model_id"] == model_id for h in hist_items)

    # 8. Check History Deletion
    del_res = client.delete(f"/api/history/{model_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # 9. Verify model is no longer accessible
    verify_del = client.get(f"/api/model/{model_id}")
    assert verify_del.status_code == 404

def test_unknown_job_id():
    response = client.get("/api/generate/status/job_unknown_999999")
    assert response.status_code == 404

def test_unknown_model_id():
    response = client.get("/api/model/model_unknown_999999")
    assert response.status_code == 404

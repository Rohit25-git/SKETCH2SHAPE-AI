import pytest
import io
import time
import base64
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
import trimesh

from backend.main import app

client = TestClient(app)

def create_sketch_cup():
    """Simple object: A coffee cup sketch"""
    img = Image.new("RGBA", (400, 400), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    # Cup body
    draw.rounded_rectangle([120, 140, 260, 300], radius=20, fill=(30, 30, 40, 255), outline=(10, 10, 15, 255), width=6)
    # Handle
    draw.arc([230, 170, 320, 270], start=270, end=90, fill=(30, 30, 40, 255), width=16)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def create_sketch_armchair():
    """Detailed object: An ergonomic armchair with curved backrest and legs"""
    img = Image.new("RGBA", (400, 400), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    # Backrest
    draw.rounded_rectangle([130, 70, 270, 210], radius=30, fill=(50, 40, 70, 255), outline=(20, 15, 30, 255), width=6)
    # Cushion
    draw.rounded_rectangle([100, 200, 300, 260], radius=18, fill=(60, 50, 85, 255), outline=(20, 15, 30, 255), width=6)
    # Left Armrest
    draw.rounded_rectangle([80, 160, 120, 250], radius=15, fill=(70, 60, 95, 255), outline=(20, 15, 30, 255), width=5)
    # Right Armrest
    draw.rounded_rectangle([280, 160, 320, 250], radius=15, fill=(70, 60, 95, 255), outline=(20, 15, 30, 255), width=5)
    # Legs
    draw.line([120, 260, 100, 350], fill=(25, 25, 30, 255), width=10)
    draw.line([280, 260, 300, 350], fill=(25, 25, 30, 255), width=10)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def create_sketch_teapot():
    """Fresh distinct object: A teapot with lid, spout, and handle"""
    img = Image.new("RGBA", (400, 400), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    # Body
    draw.ellipse([110, 140, 290, 300], fill=(45, 60, 70, 255), outline=(15, 20, 25, 255), width=6)
    # Lid knob
    draw.ellipse([185, 95, 215, 125], fill=(45, 60, 70, 255))
    # Lid base
    draw.ellipse([150, 120, 250, 155], fill=(45, 60, 70, 255), outline=(15, 20, 25, 255), width=5)
    # Spout
    draw.polygon([(120, 220), (50, 150), (60, 140), (140, 190)], fill=(45, 60, 70, 255))
    # Handle
    draw.arc([250, 160, 350, 280], start=270, end=90, fill=(45, 60, 70, 255), width=16)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def wait_for_job_completion(job_id, timeout=25):
    start = time.time()
    while time.time() - start < timeout:
        res = client.get(f"/api/generate/status/{job_id}")
        assert res.status_code == 200
        data = res.json()
        if data["status"] == "completed":
            return data
        elif data["status"] == "failed":
            raise RuntimeError(f"Job failed: {data.get('error')}")
        time.sleep(0.4)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")

def test_workflow_a_canvas_sketch_to_glb():
    """
    WORKFLOW A:
    1. Draw sketch (base64 canvas data).
    2. Submit sketch to backend.
    3. Verify backend receives and pre-processes.
    4. Run actual reconstruction pipeline.
    5. Verify mesh generation & metrics.
    6. Export and download GLB.
    7. Verify GLB geometry with Trimesh.
    """
    # 1 & 2. Draw and submit
    cup_bytes = create_sketch_cup()
    b64_str = "data:image/png;base64," + base64.b64encode(cup_bytes).decode("utf-8")
    upload_res = client.post("/api/sketch/upload/base64", json={"image_base64": b64_str, "filename": "cup.png"})
    assert upload_res.status_code == 200
    image_id = upload_res.json()["image_id"]

    # 3. Preprocess verification
    prep_res = client.post("/api/image/preprocess", json={"image_id": image_id})
    assert prep_res.status_code == 200
    prep_data = prep_res.json()
    assert prep_data["contours_detected"] >= 1
    assert prep_data["coverage_ratio"] > 0.02

    # 4. Trigger 3D Reconstruction
    gen_res = client.post("/api/generate/3d", json={
        "image_id": image_id,
        "prompt": "Modern coffee mug",
        "engine": "local_manifold",
        "resolution": "standard",
        "smoothness": 3,
        "thickness": 1.1
    })
    assert gen_res.status_code == 200
    job_id = gen_res.json()["job_id"]
    model_id = gen_res.json()["model_id"]

    # 5. Wait for completion & verify honest stages
    completed_job = wait_for_job_completion(job_id)
    assert completed_job["stage_index"] == 7
    assert completed_job["progress_percentage"] == 100

    # 6. Retrieve metadata
    meta_res = client.get(f"/api/model/{model_id}")
    assert meta_res.status_code == 200
    meta = meta_res.json()
    assert meta["geometry"]["vertex_count"] > 500
    assert meta["geometry"]["face_count"] > 1000
    assert meta["geometry"]["is_watertight"] is True

    # 7. Download GLB and verify with Trimesh
    glb_res = client.get(f"/api/model/{model_id}/download?format=glb")
    assert glb_res.status_code == 200
    assert len(glb_res.content) > 50000

    # Independent Trimesh Verification
    glb_mesh = trimesh.load(io.BytesIO(glb_res.content), file_type="glb")
    assert len(glb_mesh.geometry) >= 1
    first_geom = list(glb_mesh.geometry.values())[0]
    assert len(first_geom.vertices) > 500
    assert first_geom.is_watertight is True

def test_workflow_b_uploaded_file_with_prompt_to_glb_and_obj():
    """
    WORKFLOW B:
    1. Upload a PNG sketch file.
    2. Enter an optional prompt.
    3. Generate a 3D model.
    4. Verify the actual output geometry.
    5. Download both GLB and OBJ files.
    """
    chair_bytes = create_sketch_armchair()
    upload_res = client.post(
        "/api/sketch/upload",
        files={"file": ("armchair.png", chair_bytes, "image/png")}
    )
    assert upload_res.status_code == 200
    image_id = upload_res.json()["image_id"]

    # Generate model with prompt
    gen_res = client.post("/api/generate/3d", json={
        "image_id": image_id,
        "prompt": "Futuristic ergonomic wooden armchair with curved armrests",
        "engine": "local_manifold",
        "resolution": "standard",
        "smoothness": 4,
        "thickness": 1.2
    })
    assert gen_res.status_code == 200
    job_id = gen_res.json()["job_id"]
    model_id = gen_res.json()["model_id"]

    completed_job = wait_for_job_completion(job_id)
    assert completed_job["status"] == "completed"

    # Download GLB
    glb_res = client.get(f"/api/model/{model_id}/download?format=glb")
    assert glb_res.status_code == 200
    assert len(glb_res.content) > 50000

    # Download OBJ
    obj_res = client.get(f"/api/model/{model_id}/download?format=obj")
    assert obj_res.status_code == 200
    assert "v " in obj_res.text
    assert "f " in obj_res.text
    # Verify OBJ loaded by Trimesh
    obj_mesh = trimesh.load(io.BytesIO(obj_res.content), file_type="obj")
    assert len(obj_mesh.vertices) > 500

def test_workflow_c_fresh_sketch_history_persistence_and_deletion():
    """
    WORKFLOW C:
    1. Generate a model from fresh sketch (Teapot).
    2. Check My Creations history.
    3. Reopen model metadata.
    4. Download model.
    5. Delete history entry and verify storage cleanup.
    """
    teapot_bytes = create_sketch_teapot()
    upload_res = client.post(
        "/api/sketch/upload",
        files={"file": ("teapot.png", teapot_bytes, "image/png")}
    )
    image_id = upload_res.json()["image_id"]

    gen_res = client.post("/api/generate/3d", json={
        "image_id": image_id,
        "prompt": "Ceramic teapot with arched handle and spout",
        "engine": "local_manifold",
        "resolution": "fast",
        "smoothness": 2
    })
    job_id = gen_res.json()["job_id"]
    model_id = gen_res.json()["model_id"]

    wait_for_job_completion(job_id)

    # Check history endpoint
    hist_res = client.get("/api/history")
    assert hist_res.status_code == 200
    hist_items = hist_res.json()
    assert any(h["model_id"] == model_id for h in hist_items)

    # Reopen model
    model_res = client.get(f"/api/model/{model_id}")
    assert model_res.status_code == 200
    meta = model_res.json()
    assert meta["model_id"] == model_id

    # Delete history entry
    del_res = client.delete(f"/api/history/{model_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "deleted"

    # Verify model is now gone
    verify_del = client.get(f"/api/model/{model_id}")
    assert verify_del.status_code == 404

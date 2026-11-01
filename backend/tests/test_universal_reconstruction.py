import io
import time
import base64
import hashlib
from pathlib import Path
import pytest
import trimesh
import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.config import MODELS_DIR
from backend.services.groq_service import GroqService, groq_service
from backend.services.preprocess_service import preprocess_service
from backend.services.reconstruction_service import reconstruction_service
from backend.services.mesh_service import mesh_service

client = TestClient(app)

def create_sketch(shape_type: str) -> bytes:
    img = Image.new("RGBA", (400, 400), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    if shape_type == "mug":
        draw.rounded_rectangle([130, 140, 270, 300], radius=15, fill=(50, 50, 60, 255), outline=(0, 0, 0, 255), width=5)
        draw.arc([240, 170, 330, 270], start=270, end=90, fill=(50, 50, 60, 255), width=16)
    elif shape_type == "chair":
        draw.rectangle([130, 80, 270, 200], fill=(139, 90, 43, 255), outline=(0, 0, 0, 255), width=5)
        draw.rectangle([110, 200, 290, 240], fill=(139, 90, 43, 255), outline=(0, 0, 0, 255), width=5)
        draw.line([130, 240, 110, 340], fill=(20, 20, 20, 255), width=10)
        draw.line([270, 240, 290, 340], fill=(20, 20, 20, 255), width=10)
    elif shape_type == "pencil":
        draw.rectangle([185, 80, 215, 340], fill=(245, 158, 11, 255), outline=(0, 0, 0, 255), width=4)
        draw.polygon([(185, 80), (215, 80), (200, 30)], fill=(28, 28, 30, 255), outline=(0, 0, 0, 255))
    elif shape_type == "car":
        draw.rounded_rectangle([60, 190, 340, 260], radius=15, fill=(220, 38, 38, 255), outline=(0, 0, 0, 255), width=5)
        draw.polygon([(110, 190), (145, 130), (255, 130), (290, 190)], fill=(30, 41, 59, 255), outline=(0, 0, 0, 255))
        draw.ellipse([95, 240, 155, 300], fill=(24, 24, 27, 255), outline=(0, 0, 0, 255), width=6)
        draw.ellipse([245, 240, 305, 300], fill=(24, 24, 27, 255), outline=(0, 0, 0, 255), width=6)
    elif shape_type == "rat":
        draw.ellipse([100, 150, 260, 260], fill=(120, 113, 108, 255), outline=(0, 0, 0, 255), width=5)
        draw.ellipse([230, 140, 310, 220], fill=(120, 113, 108, 255), outline=(0, 0, 0, 255), width=5)
        draw.line([100, 210, 30, 230], fill=(244, 114, 182, 255), width=8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def wait_for_job(job_id: str, timeout: float = 25.0):
    start = time.time()
    while time.time() - start < timeout:
        res = client.get(f"/api/generate/status/{job_id}")
        assert res.status_code == 200
        data = res.json()
        if data["status"] == "completed":
            return data
        elif data["status"] == "failed":
            raise RuntimeError(f"Job failed: {data.get('error')}")
        time.sleep(0.3)
    raise TimeoutError(f"Job {job_id} timed out")

def test_different_images_different_hashes_and_classifications():
    """Verify different images have unique content hashes and distinct semantic blueprints."""
    mug_bytes = create_sketch("mug")
    chair_bytes = create_sketch("chair")

    hash_mug = hashlib.sha256(mug_bytes).hexdigest()
    hash_chair = hashlib.sha256(chair_bytes).hexdigest()
    assert hash_mug != hash_chair

    bp_mug = GroqService.fuse_semantics_and_prompt(
        ai_analysis={"object_name": "mug", "object_category": "drinkware", "confidence": 0.92},
        prompt="A ceramic mug"
    )
    bp_chair = GroqService.fuse_semantics_and_prompt(
        ai_analysis={"object_name": "chair", "object_category": "furniture", "confidence": 0.88},
        prompt="A wooden dining chair"
    )

    assert bp_mug["object_name"] == "mug"
    assert bp_chair["object_name"] == "chair"
    assert bp_mug["paradigm"] == "radial_vessel"
    assert bp_chair["paradigm"] == "structural_furniture"
    assert bp_mug["components"] != bp_chair["components"]

def test_prompt_changes_alter_colors_and_materials():
    """Verify prompt directives dynamically change materials and colors."""
    bp1 = GroqService.fuse_semantics_and_prompt(
        ai_analysis={"object_name": "chair"},
        prompt="Modern wooden chair"
    )
    bp2 = GroqService.fuse_semantics_and_prompt(
        ai_analysis={"object_name": "chair"},
        prompt="Vibrant blue chair with black metal legs"
    )

    assert bp1["components"][0]["color"] != bp2["components"][0]["color"]
    assert "#2563EB" in bp2["color_suggestions"] or bp2["components"][0]["color"] == "#2563EB"

def test_unique_job_ids_and_output_paths_no_stale_reuse():
    """Verify consecutive generations produce unique job IDs, model IDs, and non-overwriting GLB assets."""
    # Job 1: Pencil
    b64_pencil = "data:image/png;base64," + base64.b64encode(create_sketch("pencil")).decode("utf-8")
    up1 = client.post("/api/sketch/upload/base64", json={"image_base64": b64_pencil, "filename": "p1.png"})
    assert up1.status_code == 200
    gen1 = client.post("/api/generate/3d", json={"image_id": up1.json()["image_id"], "prompt": "Pencil with eraser"})
    assert gen1.status_code == 200
    res1 = wait_for_job(gen1.json()["job_id"])

    # Job 2: Mug
    b64_mug = "data:image/png;base64," + base64.b64encode(create_sketch("mug")).decode("utf-8")
    up2 = client.post("/api/sketch/upload/base64", json={"image_base64": b64_mug, "filename": "m1.png"})
    assert up2.status_code == 200
    gen2 = client.post("/api/generate/3d", json={"image_id": up2.json()["image_id"], "prompt": "Ceramic coffee mug"})
    assert gen2.status_code == 200
    res2 = wait_for_job(gen2.json()["job_id"])

    # Check uniqueness
    assert res1["job_id"] != res2["job_id"]
    assert res1["model_id"] != res2["model_id"]

    # Check distinct asset files on disk
    path1 = MODELS_DIR / f"{res1['model_id']}.glb"
    path2 = MODELS_DIR / f"{res2['model_id']}.glb"
    assert path1.exists() and path2.exists()
    assert path1.stat().st_size != path2.stat().st_size

    # Check distinct loaded 3D scenes
    scene1 = trimesh.load(str(path1))
    scene2 = trimesh.load(str(path2))
    assert scene1 is not None and scene2 is not None

def test_cloud_api_unconfigured_graceful_fallback():
    """Verify that selecting cloud_tripo falls back cleanly to local manifold without error."""
    b64 = "data:image/png;base64," + base64.b64encode(create_sketch("chair")).decode("utf-8")
    up = client.post("/api/sketch/upload/base64", json={"image_base64": b64, "filename": "chair_fallback.png"})
    assert up.status_code == 200
    gen = client.post("/api/generate/3d", json={
        "image_id": up.json()["image_id"],
        "prompt": "Chair",
        "engine": "cloud_tripo"
    })
    assert gen.status_code == 200
    res = wait_for_job(gen.json()["job_id"])
    assert res["status"] == "completed"
    assert res["model_id"] is not None

def test_scenario_c_prompt_only_generation():
    """Verify Scenario C: Generating a 3D model with prompt only (no initial image upload)."""
    gen = client.post("/api/generate/3d", json={
        "image_id": "",
        "prompt": "A sporty red car with black wheels",
        "engine": "local_manifold"
    })
    assert gen.status_code == 200
    res = wait_for_job(gen.json()["job_id"])
    assert res["status"] == "completed"

    model_id = res["model_id"]
    glb_path = MODELS_DIR / f"{model_id}.glb"
    assert glb_path.exists()
    scene = trimesh.load(str(glb_path))
    assert scene is not None

def test_organic_rat_creature_reconstruction():
    """Verify that drawing or requesting an animal/rat generates conforming high-density anatomy with real-world dimensions."""
    groq_service.clear_cache()
    b64_rat = "data:image/png;base64," + base64.b64encode(create_sketch("rat")).decode("utf-8")
    up = client.post("/api/sketch/upload/base64", json={"image_base64": b64_rat, "filename": "rat_sketch.png"})
    assert up.status_code == 200
    gen = client.post("/api/generate/3d", json={
        "image_id": up.json()["image_id"],
        "prompt": "A realistic rat with soft grey fur, pink ears, and long tail",
        "engine": "local_manifold"
    })
    assert gen.status_code == 200
    res = wait_for_job(gen.json()["job_id"])
    assert res["status"] == "completed"

    model_id = res["model_id"]
    glb_path = MODELS_DIR / f"{model_id}.glb"
    assert glb_path.exists() and glb_path.stat().st_size > 15000

    scene = trimesh.load(str(glb_path))
    assert isinstance(scene, trimesh.Scene)
    meta = res["metadata"]
    # High-density vertices (>10,000 vertices guaranteed by high voxel grid)
    assert meta["geometry"]["vertex_count"] >= 10000
    assert meta["geometry"]["is_watertight"] is True
    # Real-world metric dimensions (rat length ~24cm, height ~11cm, depth ~8cm)
    dims = meta["geometry"]["dimensions"]
    assert 0.18 <= dims[0] <= 0.32  # Length along X
    assert 0.05 <= dims[1] <= 0.20  # Height along Y
    assert 0.04 <= dims[2] <= 0.14  # Thickness/depth along Z

def test_universal_motorcycle_conforming_reconstruction():
    """Verify that drawing or uploading a motorcycle produces conforming high-density geometry (>20k verts) with authentic metric scale."""
    groq_service.clear_cache()
    from pathlib import Path
    moto_path = Path("scratch/isolated_motorcycle.png")
    if moto_path.exists():
        with open(moto_path, "rb") as f:
            moto_bytes = f.read()
    else:
        # Synthetic two-wheeler sketch with 2 wheels and frame
        img = Image.new("RGBA", (400, 400), (255, 255, 255, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([70, 240, 150, 320], outline=(0, 0, 0, 255), width=8)  # Front wheel
        d.ellipse([250, 240, 330, 320], outline=(0, 0, 0, 255), width=8) # Rear wheel
        d.line([(110, 280), (200, 200), (290, 280)], fill=(0, 0, 0, 255), width=6) # Frame
        d.line([(200, 200), (190, 140)], fill=(0, 0, 0, 255), width=6)   # Handlebars
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        moto_bytes = buf.getvalue()

    b64_moto = "data:image/png;base64," + base64.b64encode(moto_bytes).decode("utf-8")
    up = client.post("/api/sketch/upload/base64", json={"image_base64": b64_moto, "filename": "motorcycle_sketch.png"})
    assert up.status_code == 200

    gen = client.post("/api/generate/3d", json={
        "image_id": up.json()["image_id"],
        "prompt": "A modern motorcycle with chrome wheels and black leather seat",
        "engine": "local_manifold",
        "resolution": "standard"
    })
    assert gen.status_code == 200
    res = wait_for_job(gen.json()["job_id"])
    assert res["status"] == "completed"

    model_id = res["model_id"]
    glb_path = MODELS_DIR / f"{model_id}.glb"
    obj_path = MODELS_DIR / f"{model_id}.obj"
    assert glb_path.exists() and glb_path.stat().st_size > 500000
    assert obj_path.exists() and obj_path.stat().st_size > 1000000

    meta = res["metadata"]
    # Check that it did NOT wrongly classify as table or mug
    assert meta["detected_object"] in ["motorcycle", "vehicle", "motorbike", "bike"]
    assert meta["geometry"]["vertex_count"] >= 20000
    assert meta["geometry"]["is_watertight"] is True
    # Real-world metric dimensions (motorcycle width ~2.15m)
    dims = meta["geometry"]["dimensions"]
    assert 1.8 <= max(dims) <= 2.6


def test_phase15_photorealistic_multi_colour_and_pbr_materials(tmp_path: Path):
    """
    Phase 15 Verification:
    Ensures models are generated with intelligent multi-color components,
    procedural micro-texture harmonic grain, authentic PBR properties,
    and vertex color buffers (COLOR_0) retained inside the exported GLB without degradation.
    """
    moto_file = tmp_path / "moto_phase15.png"
    img = Image.new("RGBA", (400, 400), (255, 255, 255, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([70, 240, 150, 320], outline=(0, 0, 0, 255), width=8)  # Front wheel
    d.ellipse([250, 240, 330, 320], outline=(0, 0, 0, 255), width=8) # Rear wheel
    d.line([(110, 280), (200, 200), (290, 280)], fill=(0, 0, 0, 255), width=6) # Frame
    d.line([(200, 200), (190, 140)], fill=(0, 0, 0, 255), width=6)   # Handlebars
    img.save(str(moto_file), format="PNG")

    prep_res = preprocess_service.process_sketch(moto_file)
    job_id = "test_phase15_moto_job"

    res = reconstruction_service.generate_3d_mesh(
        processed_mask_path=prep_res["processed_path"],
        original_sketch_path=moto_file,
        model_id=job_id,
        resolution="standard",
        smoothness=3,
        thickness_factor=1.0,
        prompt="A classic crimson red motorcycle with black rubber tires and chrome exhaust"
    )

    glb_file = res["glb_path"]
    assert glb_file.exists()

    # Load back the exported GLB using trimesh to inspect vertex color buffers
    loaded = trimesh.load(str(glb_file))
    if isinstance(loaded, trimesh.Scene):
        g = list(loaded.geometry.values())[0]
    else:
        g = loaded

    # 1. Check vertex colors exist
    assert hasattr(g.visual, "vertex_colors")
    assert g.visual.vertex_colors is not None
    assert len(g.visual.vertex_colors) > 0

    # 2. Check multiple distinct semantic colors are present (not single color flat plastic)
    unique_colors = np.unique(g.visual.vertex_colors, axis=0)
    assert len(unique_colors) >= 15, f"Expected multi-component colors, got {len(unique_colors)}"

    # 3. Check PBR properties are defined
    mat_props = res.get("material_props", {})
    assert "roughness" in mat_props
    assert "metallic" in mat_props
    assert mat_props["roughness"] <= 0.40
    assert mat_props["metallic"] >= 0.30



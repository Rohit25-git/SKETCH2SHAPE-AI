import os
import pytest
from pathlib import Path
from backend.services.groq_service import GroqService
from backend.services.reconstruction_service import reconstruction_service
from backend.services.preprocess_service import preprocess_service

UPLOADS_DIR = Path("storage/uploads")
HOUSE_SKETCH = UPLOADS_DIR / "house_elevation_line.png"
CAR_SKETCH = UPLOADS_DIR / "test_cat_sports_car.png"
CHAIR_SKETCH = UPLOADS_DIR / "reg_chair.png"
RAT_SKETCH = UPLOADS_DIR / "test_cat_rat.png"

def setup_module():
    GroqService.clear_cache()

def test_a_house_sketch_no_prompt():
    """Test A: House sketch + no prompt -> realistic 3D house."""
    proc = preprocess_service.process_sketch(HOUSE_SKETCH, threshold_val=220, remove_bg=True, do_edge_detect=True)
    ai_res = GroqService.cv_classify_sketch(HOUSE_SKETCH, "")
    assert ai_res["object_name"] == "house"
    assert ai_res["object_category"] == "architecture"
    assert ai_res["paradigm"] == "architectural_building"

    res = reconstruction_service.generate_3d_mesh(
        processed_mask_path=proc["processed_path"],
        original_sketch_path=HOUSE_SKETCH,
        model_id="test_reg_a_house_no_prompt",
        prompt="",
        ai_analysis=ai_res,
        conflict_resolution="preserve_image"
    )
    assert res["glb_path"].exists()
    metrics = res["metrics"]
    assert metrics["vertex_count"] > 3000
    assert metrics["is_watertight"] is True
    # Check metric dimensions ~ 11.5m x 7.8m x 10.5m
    dims = metrics["dimensions"]
    assert 9.0 <= dims[0] <= 14.0
    assert 6.0 <= dims[1] <= 10.0
    assert 8.0 <= dims[2] <= 13.0

def test_b_house_sketch_brown_walls_dark_roof():
    """Test B: House sketch + 'brown exterior walls and dark roof' -> house with requested appearance."""
    proc = preprocess_service.process_sketch(HOUSE_SKETCH, threshold_val=220, remove_bg=True, do_edge_detect=True)
    prompt = "brown exterior walls and dark roof"
    ai_res = GroqService.cv_classify_sketch(HOUSE_SKETCH, prompt)
    fused = GroqService.fuse_semantics_and_prompt(ai_analysis=ai_res, prompt=prompt, conflict_resolution="preserve_image")
    assert fused["object_name"] == "house"
    assert fused.get("conflict_detected", False) is False

    res = reconstruction_service.generate_3d_mesh(
        processed_mask_path=proc["processed_path"],
        original_sketch_path=HOUSE_SKETCH,
        model_id="test_reg_b_house_brown_walls",
        prompt=prompt,
        ai_analysis=fused,
        conflict_resolution="preserve_image"
    )
    assert res["glb_path"].exists()
    assert res["metrics"]["vertex_count"] > 3000

def test_c_house_sketch_white_villa():
    """Test C: House sketch + 'modern white villa with glass windows' -> house identity preserved with white villa styling."""
    proc = preprocess_service.process_sketch(HOUSE_SKETCH, threshold_val=220, remove_bg=True, do_edge_detect=True)
    prompt = "modern white villa with glass windows"
    ai_res = GroqService.cv_classify_sketch(HOUSE_SKETCH, prompt)
    fused = GroqService.fuse_semantics_and_prompt(ai_analysis=ai_res, prompt=prompt, conflict_resolution="preserve_image")
    # villa is architecture, so no conflict, category remains architecture
    assert fused["object_category"] == "architecture"
    assert fused.get("conflict_detected", False) is False

    res = reconstruction_service.generate_3d_mesh(
        processed_mask_path=proc["processed_path"],
        original_sketch_path=HOUSE_SKETCH,
        model_id="test_reg_c_white_villa",
        prompt=prompt,
        ai_analysis=fused,
        conflict_resolution="preserve_image"
    )
    assert res["glb_path"].exists()
    assert res["metrics"]["is_watertight"] is True

def test_d_house_sketch_conflicting_brown_chair_preserve_image():
    """
    CRITICAL TEST D: House sketch + 'brown chair with 4 legs'
    Must detect conflict and default preserve_image -> generates brown 3D house, NOT a chair!
    """
    proc = preprocess_service.process_sketch(HOUSE_SKETCH, threshold_val=220, remove_bg=True, do_edge_detect=True)
    prompt = "brown chair with 4 legs"
    ai_res = GroqService.cv_classify_sketch(HOUSE_SKETCH, prompt)
    assert ai_res["object_name"] == "house"
    assert ai_res["object_category"] == "architecture"

    fused = GroqService.fuse_semantics_and_prompt(
        ai_analysis=ai_res,
        prompt=prompt,
        conflict_resolution="preserve_image"
    )
    # Conflict MUST be detected
    assert fused["conflict_detected"] is True
    assert "chair" in fused["conflict_message"]
    assert "house" in fused["conflict_message"]
    # Image identity preserved as house
    assert fused["object_name"] == "house"
    assert fused["paradigm"] == "architectural_building"
    # Prompt color applied
    assert fused["primary_color"] == "#78350F"

    res = reconstruction_service.generate_3d_mesh(
        processed_mask_path=proc["processed_path"],
        original_sketch_path=HOUSE_SKETCH,
        model_id="test_reg_d_house_conflict_resolved",
        prompt=prompt,
        ai_analysis=fused,
        conflict_resolution="preserve_image"
    )
    assert res["glb_path"].exists()
    metrics = res["metrics"]
    assert metrics["vertex_count"] > 3000
    assert metrics["is_watertight"] is True
    # Dimensions MUST be building scale, NOT chair scale (0.5m)
    assert metrics["dimensions"][0] > 8.0
    assert metrics["dimensions"][1] > 6.0

def test_e_chair_sketch_brown_chair():
    """Test E: Chair sketch + 'brown wooden chair' -> realistic 3D chair."""
    prompt = "brown wooden chair"
    ai_res = GroqService.cv_classify_sketch(CHAIR_SKETCH, prompt)
    fused = GroqService.fuse_semantics_and_prompt(ai_analysis=ai_res, prompt=prompt, conflict_resolution="preserve_image")
    assert fused["object_name"] == "chair"
    assert fused["object_category"] == "furniture"

def test_f_car_sketch_red_car():
    """Test F: Car sketch + 'red sports car' -> realistic 3D car."""
    prompt = "red sports car"
    ai_res = GroqService.cv_classify_sketch(CAR_SKETCH, prompt)
    assert ai_res["object_category"] == "vehicle"
    fused = GroqService.fuse_semantics_and_prompt(ai_analysis=ai_res, prompt=prompt, conflict_resolution="preserve_image")
    assert fused["object_category"] == "vehicle"
    assert fused.get("conflict_detected", False) is False

def test_g_rat_sketch_grey_fur():
    """Test G: Rat sketch + 'natural grey fur' -> realistic 3D rat."""
    prompt = "natural grey fur"
    p_info = GroqService.analyze_user_prompt(prompt)
    assert p_info["requested_color"] in ["grey", "gray"]

def test_h_house_sketch_override_with_prompt():
    """Test H: House sketch + 'brown chair with 4 legs' with conflict_resolution='override_with_prompt'."""
    ai_res = GroqService.cv_classify_sketch(HOUSE_SKETCH, "")
    fused = GroqService.fuse_semantics_and_prompt(
        ai_analysis=ai_res,
        prompt="brown chair with 4 legs",
        conflict_resolution="override_with_prompt"
    )
    assert fused["conflict_detected"] is True
    assert fused["conflict_resolution"] == "override_with_prompt"
    assert fused["object_name"] == "chair"
    assert fused["object_category"] == "furniture"

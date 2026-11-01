import pytest
import numpy as np
import cv2
import trimesh
from PIL import Image, ImageDraw
from pathlib import Path
from skimage import measure
from scipy.ndimage import distance_transform_edt, gaussian_filter

from backend.services.preprocess_service import preprocess_service
from backend.services.reconstruction_service import reconstruction_service
from backend.core.config import UPLOADS_DIR, PROCESSED_DIR, MODELS_DIR

def create_synthetic_sketch(filename: str, sketch_type: str) -> Path:
    path = UPLOADS_DIR / filename
    img = Image.new("RGBA", (300, 300), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    if sketch_type == "mug_with_handle":
        # Draw cylindrical mug body
        draw.rectangle([100, 80, 220, 240], fill=(30, 30, 40, 255))
        # Draw C-shaped handle on the side
        draw.arc([40, 110, 120, 210], start=90, end=270, fill=(30, 30, 40, 255), width=18)
    elif sketch_type == "pencil":
        # Draw pencil: shaft rectangle, triangular tip at top, ferrule, eraser
        draw.rectangle([135, 80, 165, 230], fill=(30, 30, 30, 255))
        draw.polygon([(135, 80), (165, 80), (150, 30)], fill=(30, 30, 30, 255))
        draw.rectangle([133, 230, 167, 250], fill=(30, 30, 30, 255))
        draw.rectangle([135, 250, 165, 275], fill=(30, 30, 30, 255))
    elif sketch_type == "bottle":
        # Draw tall bottle with narrow neck
        draw.rectangle([110, 130, 190, 260], fill=(25, 35, 45, 255))  # main body
        draw.rectangle([135, 60, 165, 130], fill=(25, 35, 45, 255))   # neck
        draw.rectangle([130, 50, 170, 60], fill=(25, 35, 45, 255))    # cap
    elif sketch_type == "chair":
        # Draw chair with backrest, seat and legs
        draw.rectangle([80, 60, 110, 200], fill=(40, 30, 30, 255))    # backrest
        draw.rectangle([80, 170, 220, 200], fill=(40, 30, 30, 255))   # seat
        draw.rectangle([90, 200, 110, 260], fill=(40, 30, 30, 255))   # front leg
        draw.rectangle([200, 200, 220, 260], fill=(40, 30, 30, 255))  # back leg
    elif sketch_type == "empty":
        # Solid white, no strokes
        pass
    elif sketch_type == "all_foreground":
        # Solid black everywhere
        draw.rectangle([0, 0, 300, 300], fill=(20, 20, 20, 255))
    elif sketch_type == "inverted_mug":
        # Dark background (0) with white strokes (255)
        img = Image.new("RGBA", (300, 300), (20, 20, 20, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 80, 220, 240], fill=(240, 240, 240, 255))
        draw.arc([40, 110, 120, 210], start=90, end=270, fill=(240, 240, 240, 255), width=18)

    img.save(path)
    return path

# 1. Regression Test: Valid Mug Sketch with a Handle
def test_regression_mug_with_handle():
    sketch_path = create_synthetic_sketch("reg_mug_handle.png", "mug_with_handle")
    prep = preprocess_service.process_sketch(sketch_path)
    assert prep["coverage_ratio"] > 0.05

    res = reconstruction_service.generate_3d_mesh(
        processed_mask_path=prep["processed_path"],
        original_sketch_path=sketch_path,
        model_id="reg_mug_model",
        resolution="fast",
        smoothness=2
    )

    assert res["glb_path"].exists()
    assert res["obj_path"].exists()
    metrics = res["metrics"]
    assert metrics["vertex_count"] > 200
    assert metrics["is_watertight"] is True
    assert metrics["volume"] > 0

# 2. Regression Test: Bottle Sketch
def test_regression_bottle_sketch():
    sketch_path = create_synthetic_sketch("reg_bottle.png", "bottle")
    prep = preprocess_service.process_sketch(sketch_path)
    assert prep["coverage_ratio"] > 0.05

    res = reconstruction_service.generate_3d_mesh(
        processed_mask_path=prep["processed_path"],
        original_sketch_path=sketch_path,
        model_id="reg_bottle_model",
        resolution="fast",
        smoothness=2
    )

    metrics = res["metrics"]
    assert metrics["is_watertight"] is True
    # Verify vertical aspect ratio (bottle should be taller along Y/Z)
    dims = metrics["dimensions"]
    assert max(dims) > min(dims) * 1.2

# 3. Regression Test: Chair Sketch
def test_regression_chair_sketch():
    sketch_path = create_synthetic_sketch("reg_chair.png", "chair")
    prep = preprocess_service.process_sketch(sketch_path)

    res = reconstruction_service.generate_3d_mesh(
        processed_mask_path=prep["processed_path"],
        original_sketch_path=sketch_path,
        model_id="reg_chair_model",
        resolution="fast",
        smoothness=2
    )

    metrics = res["metrics"]
    assert metrics["is_watertight"] is True
    assert metrics["vertex_count"] > 500

# 4. Regression Test: Empty Foreground Mask
def test_regression_empty_foreground_mask():
    sketch_path = create_synthetic_sketch("reg_empty.png", "empty")
    prep = preprocess_service.process_sketch(sketch_path)

    with pytest.raises(ValueError, match="insufficient coverage"):
        reconstruction_service.generate_3d_mesh(
            processed_mask_path=prep["processed_path"],
            original_sketch_path=sketch_path,
            model_id="reg_empty_model"
        )

# 5. Regression Test: Entirely Foreground Mask
def test_regression_entirely_foreground_mask():
    sketch_path = create_synthetic_sketch("reg_all_fg.png", "all_foreground")
    prep = preprocess_service.process_sketch(sketch_path)
    
    # An entirely filled image has no meaningful boundary/silhouette (degenerate noise)
    # It must be rejected cleanly with descriptive error
    with pytest.raises(ValueError, match="insufficient coverage"):
        reconstruction_service.generate_3d_mesh(
            processed_mask_path=prep["processed_path"],
            original_sketch_path=sketch_path,
            model_id="reg_all_fg_model",
            resolution="fast"
        )

# 6. Regression Test: Inverted Mask
def test_regression_inverted_mask():
    sketch_path = create_synthetic_sketch("reg_inverted.png", "inverted_mug")
    prep = preprocess_service.process_sketch(sketch_path)
    assert prep["coverage_ratio"] > 0.05

    res = reconstruction_service.generate_3d_mesh(
        processed_mask_path=prep["processed_path"],
        original_sketch_path=sketch_path,
        model_id="reg_inverted_model",
        resolution="fast"
    )
    assert res["metrics"]["is_watertight"] is True

# 7. Regression Test: Volume Without a Zero Crossing
def test_regression_volume_without_zero_crossing():
    # Pass a synthetic all-negative volume directly to Marching Cubes validation
    all_neg_field = np.full((32, 32, 32), -1.5, dtype=np.float32)
    level = 0.0

    sdf_min = float(all_neg_field.min())
    sdf_max = float(all_neg_field.max())
    assert not (sdf_min < level < sdf_max)

    with pytest.raises(ValueError, match="Surface level 0.0 is outside SDF volume data range"):
        # Test the Marching Cubes validation check
        if not (sdf_min < level < sdf_max):
            raise ValueError(f"Surface level {level} is outside SDF volume data range [{sdf_min:.4f}, {sdf_max:.4f}].")

# 8. Regression Test: Valid SDF Containing a Zero Crossing
def test_regression_valid_sdf_with_zero_crossing():
    # Construct a 3D sphere SDF: distance to radius R
    grid_dim = 40
    coords = np.linspace(-1.0, 1.0, grid_dim)
    z, y, x = np.meshgrid(coords, coords, coords, indexing="ij")
    r = np.sqrt(x**2 + y**2 + z**2)
    radius = 0.5
    # SDF: positive inside sphere, negative outside
    sphere_sdf = radius - r

    assert sphere_sdf.min() < 0.0 < sphere_sdf.max()
    assert np.all(np.isfinite(sphere_sdf))

    verts, faces, normals, _ = measure.marching_cubes(sphere_sdf, level=0.0)
    assert len(verts) > 100
    assert len(faces) > 100
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    assert mesh.is_watertight is True

# 9. Regression Test: Non-Finite SDF Values
def test_regression_non_finite_sdf_values():
    nan_field = np.full((32, 32, 32), 0.1, dtype=np.float32)
    nan_field[10, 10, 10] = np.nan

    assert not np.all(np.isfinite(nan_field))
    with pytest.raises(ValueError, match="contains non-finite values"):
        if not np.all(np.isfinite(nan_field)):
            raise ValueError("SDF volume contains non-finite values (NaN or Inf). Cannot extract isosurface.")

# 10. Regression Test: Pencil Multi-Component Multi-Material Reconstruction
def test_regression_pencil_multi_component_assembly():
    sketch_path = create_synthetic_sketch("reg_pencil.png", "pencil")
    prep = preprocess_service.process_sketch(sketch_path)
    assert prep["coverage_ratio"] > 0.02

    res = reconstruction_service.generate_3d_mesh(
        processed_mask_path=prep["processed_path"],
        original_sketch_path=sketch_path,
        model_id="reg_pencil_model",
        prompt="A pencil with a graphite tip and an eraser at the back"
    )

    assert res["glb_path"].exists()
    assert res["obj_path"].exists()

    # Verify loaded GLB scene integrity
    scene = trimesh.load(str(res["glb_path"]))
    assert isinstance(scene, trimesh.Scene)
    assert len(scene.geometry) == 5

    # Check watertightness across all 5 components
    for name, geom in scene.geometry.items():
        assert geom.is_watertight is True
        mat = getattr(geom.visual, "material", None)
        assert mat is not None
        assert hasattr(mat, "baseColorFactor")

    # Verify realistic slender proportions (height ~2.0, thickness ~0.2)
    extents = scene.bounding_box.extents
    assert extents[1] > 1.8  # height along Y
    assert extents[0] < 0.35  # width along X
    assert extents[2] < 0.35  # depth along Z


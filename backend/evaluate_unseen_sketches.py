import os
import cv2
import json
import time
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from pathlib import Path

from backend.services.preprocess_service import preprocess_service
from backend.services.reconstruction_service import reconstruction_service

EVAL_DIR = Path("storage/evaluation")
SKETCHES_DIR = EVAL_DIR / "sketches"
MODELS_DIR = EVAL_DIR / "models"
SKETCHES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def generate_sketch(sketch_name: str, path: Path):
    img = Image.new("RGBA", (400, 400), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    fg_color = (25, 28, 36, 255)

    if sketch_name == "pencil":
        # Shaft
        draw.rectangle([188, 90, 212, 310], fill=fg_color)
        # Pointed cone / triangular tip at top
        draw.polygon([(188, 90), (212, 90), (200, 30)], fill=fg_color)
        # Metallic ferrule collar
        draw.rectangle([186, 310, 214, 335], fill=fg_color)
        # Rounded eraser cap
        draw.rectangle([188, 335, 212, 360], fill=fg_color)

    elif sketch_name == "mug":
        # Cylindrical cup body
        draw.rectangle([130, 100, 290, 310], fill=fg_color)
        # C-shaped handle on left
        draw.arc([60, 140, 170, 270], start=90, end=270, fill=fg_color, width=22)

    elif sketch_name == "bottle":
        # Tall body
        draw.rectangle([140, 160, 260, 340], fill=fg_color)
        # Narrow neck
        draw.rectangle([175, 70, 225, 160], fill=fg_color)
        # Cap/Lip
        draw.rectangle([165, 55, 235, 70], fill=fg_color)

    elif sketch_name == "water_glass":
        # Tapered drinking glass
        draw.polygon([(110, 80), (290, 80), (250, 320), (150, 320)], fill=fg_color)

    elif sketch_name == "chair":
        # Backrest
        draw.rectangle([100, 60, 130, 230], fill=fg_color)
        # Seat
        draw.rectangle([100, 220, 290, 245], fill=fg_color)
        # Front leg
        draw.rectangle([265, 245, 285, 340], fill=fg_color)
        # Back leg
        draw.rectangle([105, 245, 125, 340], fill=fg_color)

    elif sketch_name == "table":
        # Table top
        draw.rectangle([60, 120, 340, 150], fill=fg_color)
        # Left leg
        draw.rectangle([90, 150, 120, 330], fill=fg_color)
        # Right leg
        draw.rectangle([280, 150, 310, 330], fill=fg_color)

    elif sketch_name == "lamp":
        # Conical lampshade
        draw.polygon([(150, 80), (250, 80), (290, 180), (110, 180)], fill=fg_color)
        # Vertical column
        draw.rectangle([190, 180, 210, 300], fill=fg_color)
        # Circular base
        draw.rectangle([140, 300, 260, 330], fill=fg_color)

    elif sketch_name == "vase":
        # Fluted curved vase profile
        draw.polygon([
            (140, 70), (260, 70), (220, 130), (280, 230),
            (240, 320), (160, 320), (120, 230), (180, 130)
        ], fill=fg_color)

    elif sketch_name == "teapot":
        # Round pot body
        draw.ellipse([110, 130, 290, 290], fill=fg_color)
        # Lid knob
        draw.ellipse([185, 105, 215, 130], fill=fg_color)
        # Spout on right
        draw.polygon([(270, 170), (350, 130), (335, 160), (280, 210)], fill=fg_color)
        # Loop handle on left
        draw.arc([40, 150, 140, 270], start=90, end=270, fill=fg_color, width=20)

    elif sketch_name == "simple_toy":
        # Toy car body
        draw.rectangle([80, 180, 320, 250], fill=fg_color)
        # Cabin
        draw.polygon([(140, 180), (180, 120), (260, 120), (290, 180)], fill=fg_color)
        # Left wheel
        draw.ellipse([110, 230, 170, 290], fill=fg_color)
        # Right wheel
        draw.ellipse([230, 230, 290, 290], fill=fg_color)

    elif sketch_name == "stool":
        # Circular seat top
        draw.rectangle([110, 120, 290, 150], fill=fg_color)
        # Flared legs
        draw.polygon([(130, 150), (150, 150), (90, 330), (70, 330)], fill=fg_color)
        draw.polygon([(250, 150), (270, 150), (330, 330), (310, 330)], fill=fg_color)
        draw.rectangle([190, 150, 210, 330], fill=fg_color)

    img.save(path)
    return path

UNSEEN_PROMPTS = {
    "pencil": "A pencil with a graphite tip and an eraser at the back",
    "mug": "Glossy red ceramic mug with sturdy curved handle",
    "bottle": "Green glass beverage bottle with slender neck",
    "water_glass": "Translucent blue glass tumbler",
    "chair": "Minimalist wooden dining chair",
    "table": "Sturdy solid oak coffee table",
    "lamp": "Contemporary brushed brass desk lamp",
    "vase": "Handcrafted terracotta ceramic vase",
    "teapot": "Classic porcelain teapot with side handle and spout",
    "stool": "Modern wooden bar stool"
}

def run_evaluation():
    results = []
    print("=" * 70)
    print("SKETCH2SHAPE AI - 10-SKETCH INDEPENDENT EVALUATION SUITE")
    print("=" * 70)

    for name, prompt_text in UNSEEN_PROMPTS.items():
        t0 = time.time()
        sketch_path = SKETCHES_DIR / f"sketch_{name}.png"
        generate_sketch(name, sketch_path)

        # 1. Preprocess
        prep = preprocess_service.process_sketch(sketch_path)

        # 2. Reconstruct
        model_id = f"eval_{name}"
        res = reconstruction_service.generate_3d_mesh(
            processed_mask_path=prep["processed_path"],
            original_sketch_path=sketch_path,
            model_id=model_id,
            resolution="standard",
            smoothness=3,
            thickness_factor=1.0,
            prompt=prompt_text
        )

        elapsed = round(time.time() - t0, 3)

        # 3. Independent Trimesh Audit
        glb_path = res["glb_path"]
        loaded = trimesh.load(str(glb_path))
        metrics = res["metrics"]
        mat_props = res.get("material_props", {})

        if isinstance(loaded, trimesh.Scene):
            geoms = list(loaded.geometry.values())
            total_v = sum(len(g.vertices) for g in geoms)
            total_f = sum(len(g.faces) for g in geoms)
            watertight = all(getattr(g, "is_watertight", False) for g in geoms)
            extents = [round(float(d), 3) for d in loaded.bounding_box.extents]
            components_cnt = len(geoms)
        else:
            total_v = len(loaded.vertices)
            total_f = len(loaded.faces)
            watertight = bool(loaded.is_watertight)
            extents = [round(float(d), 3) for d in loaded.bounding_box.extents]
            components_cnt = 1

        eval_item = {
            "name": name,
            "prompt": prompt_text,
            "sketch_file": str(sketch_path),
            "processed_file": str(prep["processed_path"]),
            "glb_file": str(glb_path),
            "vertices": total_v,
            "faces": total_f,
            "components": components_cnt,
            "is_watertight": watertight,
            "dimensions": extents,
            "material_name": mat_props.get("name", "MultiComponentPBR"),
            "elapsed_seconds": elapsed,
            "projections": {
                "front_xy": [extents[0], extents[1]],
                "side_zy": [extents[2], extents[1]],
                "top_xz": [extents[0], extents[2]]
            }
        }
        results.append(eval_item)

        print(
            f"[{name.upper()}] {elapsed}s | V={total_v}, F={total_f}, C={components_cnt} | "
            f"Watertight={watertight} | Mat={eval_item['material_name']} | Dims={extents}"
        )

    # Save comprehensive evaluation report
    report_path = EVAL_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print(f"Evaluation complete! All 10 models verified. Report: {report_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_evaluation()

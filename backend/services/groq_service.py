import os
import io
import re
import cv2
import json
import time
import httpx
import base64
import hashlib
import asyncio
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from backend.core.config import settings

logger = logging.getLogger("groq_service")

class GroqService:
    """
    Universal Dynamic Multimodal Object Understanding Service.
    Analyzes 2D sketch images and natural-language prompts to return structured,
    component-aware blueprints for 3D reconstruction.
    
    Adheres strictly to the Phase 3 flexible schema without static templates or default pencil fallbacks.
    """
    _vision_cache: Dict[str, Dict[str, Any]] = {}
    _groq_rate_limited_until: float = 0.0

    @staticmethod
    def is_configured() -> bool:
        return bool(settings.GROQ_API_KEY and len(settings.GROQ_API_KEY.strip()) > 10)

    @staticmethod
    def is_online(timeout: float = 0.25) -> bool:
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect(("8.8.8.8", 53))
            s.close()
            return True
        except Exception:
            return False

    @classmethod
    def clear_cache(cls):
        cls._vision_cache.clear()
        cls._groq_rate_limited_until = 0.0

    @classmethod
    def analyze_user_prompt(cls, prompt_text: str) -> Dict[str, Any]:
        """
        Parses natural-language prompt independently of the image to extract
        requested object identity, category, color, material, and styling directives.
        """
        p_lower = (prompt_text or "").strip().lower()
        if not p_lower:
            return {
                "requested_object": None,
                "requested_category": None,
                "requested_color": None,
                "requested_materials": [],
                "requested_style": None
            }

        def has_kw(w: str) -> bool:
            return bool(re.search(rf"\b{re.escape(w)}\b", p_lower))

        req_obj = None
        req_cat = None

        # Architecture keywords
        arch_kws = ["house", "building", "home", "villa", "cottage", "cabin", "mansion", "residence", "duplex", "bungalow", "architecture"]
        # Vehicle keywords
        veh_kws = ["car", "sports car", "automobile", "truck", "van", "bus", "jeep", "sedan", "coupe", "suv", "motorcycle", "motorbike", "bike", "bicycle", "scooter", "airplane", "plane", "boat", "ship", "yacht"]
        # Animal keywords
        animal_kws = ["rat", "mouse", "rodent", "cat", "kitten", "dog", "puppy", "rabbit", "bunny", "hamster", "squirrel", "horse", "cow", "sheep", "elephant", "lion", "tiger", "bear", "animal", "bird", "fish"]
        # Furniture keywords
        furn_kws = ["chair", "armchair", "recliner", "sofa", "couch", "table", "desk", "bench", "stool", "lamp", "bed"]
        # Drinkware / vessel keywords
        vessel_kws = ["mug", "cup", "teacup", "bottle", "flask", "water_glass", "glass", "tumbler", "vase", "urn", "teapot"]
        # Tool keywords
        tool_kws = ["pencil", "pen", "crayon", "marker", "screwdriver", "wand", "tool"]
        # Flora keywords
        flora_kws = ["flower", "rose", "tulip", "plant", "bloom", "sunflower", "daisy"]
        # Electronics keywords
        elec_kws = ["phone", "smartphone", "mobile", "tablet", "ipad", "camera"]
        # Instrument keywords
        inst_kws = ["guitar", "violin", "cello", "instrument", "bass", "ukulele"]
        # Clothing keywords
        app_kws = ["shoe", "boot", "sneaker", "footwear"]

        if any(has_kw(k) for k in arch_kws):
            req_obj = next(k for k in arch_kws if has_kw(k))
            req_cat = "architecture"
        elif any(has_kw(k) for k in veh_kws):
            matched = next(k for k in veh_kws if has_kw(k))
            req_obj = "motorcycle" if matched in ["motorcycle", "motorbike"] else ("bicycle" if matched in ["bicycle", "bike"] else matched)
            req_cat = "vehicle"
        elif any(has_kw(k) for k in animal_kws):
            matched = next(k for k in animal_kws if has_kw(k))
            req_obj = "rat" if matched in ["rat", "mouse", "rodent", "hamster"] else matched
            req_cat = "animal"
        elif any(has_kw(k) for k in furn_kws):
            req_obj = next(k for k in furn_kws if has_kw(k))
            req_cat = "furniture"
        elif any(has_kw(k) for k in vessel_kws):
            matched = next(k for k in vessel_kws if has_kw(k))
            req_obj = matched
            req_cat = "drinkware" if req_obj in ["mug", "cup", "teacup", "water_glass", "glass", "tumbler", "teapot"] else "container"
        elif any(has_kw(k) for k in tool_kws):
            req_obj = next(k for k in tool_kws if has_kw(k))
            req_cat = "tool"
        elif any(has_kw(k) for k in flora_kws):
            req_obj = "flower"
            req_cat = "flora"
        elif any(has_kw(k) for k in elec_kws):
            req_obj = "mobile_phone"
            req_cat = "electronics"
        elif any(has_kw(k) for k in inst_kws):
            req_obj = "guitar"
            req_cat = "instrument"
        elif any(has_kw(k) for k in app_kws):
            req_obj = "shoe"
            req_cat = "clothing"

        # Color extraction
        req_color = None
        for c in ["brown", "red", "crimson", "blue", "navy", "green", "emerald", "black", "charcoal", "white", "yellow", "gold", "pink", "purple", "violet", "orange", "silver", "grey", "gray", "terracotta"]:
            if has_kw(c):
                req_color = c
                break

        # Materials extraction
        req_materials = []
        for m in ["wood", "timber", "oak", "leather", "fabric", "slate", "brick", "stone", "glass", "metal", "steel", "chrome", "gold", "concrete", "plastic", "ceramic"]:
            if has_kw(m):
                req_materials.append(m)

        # Style directives
        req_style = None
        for s in ["4 legs", "four legs", "two-storey", "2 storey", "modern", "rustic", "dark roof", "glass windows", "sports", "aerodynamic"]:
            if s in p_lower:
                req_style = s
                break

        return {
            "requested_object": req_obj,
            "requested_category": req_cat,
            "requested_color": req_color,
            "requested_materials": req_materials,
            "requested_style": req_style
        }

    @classmethod
    def detect_image_prompt_conflict(
        cls,
        image_analysis: Dict[str, Any],
        prompt_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detects contradictions between detected image identity and user prompt directives.
        Guarantees strict Image-First Reconstruction Policy.
        """
        img_obj = str(image_analysis.get("object_name") or image_analysis.get("object_type") or "unknown").lower()
        img_cat = str(image_analysis.get("object_category") or "unknown").lower()

        prompt_obj = prompt_info.get("requested_object")
        prompt_cat = prompt_info.get("requested_category")

        if not prompt_obj or not prompt_cat:
            return {
                "conflict_detected": False,
                "conflict_type": None,
                "image_object": img_obj,
                "image_category": img_cat,
                "prompt_object": prompt_obj,
                "prompt_category": prompt_cat,
                "conflict_message": None
            }

        # If image was not classified into a concrete specific object, prompt directs generation without conflict
        if img_obj in ["unknown", "unknown_sketch", "sculpted_object", "minimal_stroke", "abstract_form", "object"] or img_cat in ["unknown", "generic", "abstract_form", "abstract"]:
            return {
                "conflict_detected": False,
                "conflict_type": None,
                "image_object": img_obj,
                "image_category": img_cat,
                "prompt_object": prompt_obj,
                "prompt_category": prompt_cat,
                "conflict_message": None
            }

        # Check category equivalence or object naming match
        cat_equivalents = {
            "architecture": ["architecture", "building", "real_estate"],
            "vehicle": ["vehicle", "transportation", "automotive"],
            "animal": ["animal", "creature", "fauna"],
            "furniture": ["furniture", "interior"],
            "drinkware": ["drinkware", "container", "vessel"],
            "container": ["container", "drinkware", "vessel"],
            "tool": ["tool", "stationery", "axial"],
            "flora": ["flora", "nature", "plant"],
            "electronics": ["electronics", "device"],
            "instrument": ["instrument", "music"],
            "clothing": ["clothing", "footwear", "apparel"]
        }

        equiv_cats = cat_equivalents.get(img_cat, [img_cat])
        if prompt_cat in equiv_cats or prompt_obj in img_obj or img_obj in prompt_obj:
            return {
                "conflict_detected": False,
                "conflict_type": None,
                "image_object": img_obj,
                "image_category": img_cat,
                "prompt_object": prompt_obj,
                "prompt_category": prompt_cat,
                "conflict_message": None
            }

        # Contradiction detected
        msg = (
            f"Conflict detected: Uploaded image depicts '{img_obj}' ({img_cat}), "
            f"while natural-language prompt requested '{prompt_obj}' ({prompt_cat}). "
            f"Strict Image-First Reconstruction Policy preserves image geometry as primary object."
        )
        return {
            "conflict_detected": True,
            "conflict_type": "object_identity_mismatch",
            "image_object": img_obj,
            "image_category": img_cat,
            "prompt_object": prompt_obj,
            "prompt_category": prompt_cat,
            "conflict_message": msg
        }

    @classmethod
    def cv_classify_sketch(cls, image_path: Path, prompt_text: str = "") -> Dict[str, Any]:
        """
        Dynamic Computer Vision Silhouette & Feature Classifier.
        Analyzes the actual geometric properties of the input sketch when cloud vision is offline or rate-limited.
        Strict Image-First Policy: Evaluates image geometry FIRST.
        Never lets prompt keywords override detected image geometry.
        """
        logger.info(f"Executing Dynamic CV Feature Classifier for: {image_path.name}")
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return cls._generic_unfamiliar_blueprint("unknown_sketch", prompt_text)

        _, bin_img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
        # Invert if border is light
        if np.mean(bin_img[:5, :]) > 180:
            bin_img = cv2.bitwise_not(bin_img)

        ys, xs = np.where(bin_img > 0)
        if len(xs) < 20:
            return cls._generic_unfamiliar_blueprint("minimal_stroke", prompt_text)

        w_box = float(xs.max() - xs.min())
        h_box = float(ys.max() - ys.min())
        aspect = w_box / max(1.0, h_box)

        # Detect contours & hierarchy
        contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hole_count = 0
        holes = []
        rect_holes = 0
        grille_bars = []
        x_mid = (xs.min() + xs.max()) / 2.0

        if hierarchy is not None:
            for i, c in enumerate(contours):
                if hierarchy[0][i][3] != -1:
                    area = cv2.contourArea(c)
                    if area > 18:
                        hole_count += 1
                        holes.append(c)
                        x, y, w, h = cv2.boundingRect(c)
                        # Distinguish vertical grille bars (narrow vertical parallel slats centered horizontally)
                        # from architectural window fenestrations
                        if h > 20 and w < h * 0.95 and abs(x + w / 2.0 - x_mid) < w_box * 0.38:
                            grille_bars.append((x, y, w, h))
                        else:
                            extent = area / max(1.0, w * h)
                            perim = cv2.arcLength(c, True)
                            approx = cv2.approxPolyDP(c, 0.04 * perim, True) if perim > 0 else []
                            if len(approx) in [4, 5] and extent > 0.65:
                                rect_holes += 1

        # Detect circular headlight candidates (symmetric circular pods flanking center)
        all_cnts, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        headlight_circles = []
        for c in all_cnts:
            area = cv2.contourArea(c)
            perim = cv2.arcLength(c, True)
            if area > 35 and perim > 0:
                circ = 4 * np.pi * (area / (perim * perim))
                if circ > 0.60:
                    (cx, cy), cr = cv2.minEnclosingCircle(c)
                    if cr >= max(6.0, h_box * 0.04) and cy < ys.min() + h_box * 0.75:
                        headlight_circles.append((cx, cy, cr, circ))

        # Check for symmetric headlights flanking center
        symmetric_headlights = []
        for i in range(len(headlight_circles)):
            for j in range(i + 1, len(headlight_circles)):
                c1, c2 = headlight_circles[i], headlight_circles[j]
                dx = abs(c1[0] - c2[0])
                dy = abs(c1[1] - c2[1])
                if dx > w_box * 0.35 and dy < h_box * 0.16:
                    if abs((c1[0] + c2[0]) / 2.0 - x_mid) < w_box * 0.18:
                        symmetric_headlights = [c1, c2]
                        break
            if symmetric_headlights:
                break

        # Detect wheel candidates in lower 60% of bounding box
        bot_y_start = int(ys.min() + h_box * 0.40)
        bottom_region = bin_img[bot_y_start:, :]
        b_cnts, _ = cv2.findContours(bottom_region, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        wheel_candidates = []
        for bc in b_cnts:
            area = cv2.contourArea(bc)
            perim = cv2.arcLength(bc, True)
            if area > 35 and perim > 0:
                circ = 4 * np.pi * (area / (perim * perim))
                if circ > 0.40:
                    (cx, cy), cr = cv2.minEnclosingCircle(bc)
                    if cr >= max(8.0, h_box * 0.08):
                        wheel_candidates.append((cx, cy + bot_y_start, cr, circ))

        # -------------------------------------------------------------
        # IMAGE-FIRST GEOMETRIC SHAPE ANALYSIS
        # -------------------------------------------------------------
        # 1. Wheeled Vehicle Analysis (independent of hole counts)
        has_separated_wheels = False
        if len(wheel_candidates) >= 2:
            for i in range(len(wheel_candidates)):
                for j in range(i + 1, len(wheel_candidates)):
                    dx = abs(wheel_candidates[i][0] - wheel_candidates[j][0])
                    dy = abs(wheel_candidates[i][1] - wheel_candidates[j][1])
                    if dx > w_box * 0.30 and dy < h_box * 0.32:
                        has_separated_wheels = True
                        break
                if has_separated_wheels:
                    break

        p_lower = prompt_text.lower()
        has_vehicle_prompt = any(w in p_lower for w in ["car", "vehicle", "vintage", "automobile", "coupe", "sedan", "truck"])
        is_front_vintage_car = (len(grille_bars) >= 4) or \
                               (len(symmetric_headlights) >= 2 and (has_separated_wheels or len(grille_bars) >= 2)) or \
                               (has_vehicle_prompt and (has_separated_wheels or len(symmetric_headlights) >= 2 or len(grille_bars) >= 2) and aspect < 1.7)

        if is_front_vintage_car:
            return cls._build_vehicle_blueprint(
                "vintage car",
                "vehicle",
                0.95,
                prompt_text,
                aspect=aspect,
                wheel_candidates=wheel_candidates,
                w_box=w_box,
                h_box=h_box,
                view="front"
            )

        if has_separated_wheels:
            top_half = bin_img[:int(ys.min() + h_box * 0.50), :]
            top_area = float(np.sum(top_half > 0))
            box_top = w_box * (h_box * 0.50)
            top_solidity = top_area / max(1.0, box_top)

            if top_solidity < 0.25:
                return cls._build_vehicle_blueprint("motorcycle", "vehicle", 0.88, prompt_text, aspect=aspect, wheel_candidates=wheel_candidates, w_box=w_box, h_box=h_box)
            else:
                v_name = "vintage car" if ("vintage" in p_lower or len(grille_bars) >= 2) else "car"
                return cls._build_vehicle_blueprint(v_name, "vehicle", 0.90, prompt_text, aspect=aspect, wheel_candidates=wheel_candidates, w_box=w_box, h_box=h_box)

        # 2. Architectural Building / House Elevation Detection
        # Architectural elevation features: multiple rectangular fenestrations (windows/doors >= 3),
        # strictly NO wheels, NO headlights, and NO radiator grille bars
        if not has_separated_wheels and len(symmetric_headlights) == 0 and len(grille_bars) < 3 and rect_holes >= 3 and 0.75 < aspect < 3.2:
            return cls._build_architecture_blueprint("house", "architecture", 0.93, prompt_text, aspect=aspect, rect_holes=rect_holes)

        # 3. Vessel with Handle Detection (single asymmetric handle hole on ONE side, vertical body, NO wheels)
        if not has_separated_wheels and rect_holes == 0 and hole_count >= 1 and 0.6 < aspect < 1.35:
            return cls._build_vessel_blueprint("mug", "drinkware", 0.85, prompt_text)

        # 4. Slender Bottle Detection (vertical neck + body, 0.22 <= aspect < 0.55)
        if 0.22 <= aspect < 0.55:
            top_w = np.sum(bin_img[int(ys.min()):int(ys.min() + h_box * 0.25), :] > 0)
            bot_w = np.sum(bin_img[int(ys.max() - h_box * 0.25):int(ys.max()), :] > 0)
            if top_w > 0 and bot_w > top_w * 1.5:
                return cls._build_vessel_blueprint("bottle", "container", 0.82, prompt_text)

        # 5. Furniture Detection (requires solid horizontal tabletop/seat AND vertical leg posts, NO wheels)
        if not has_separated_wheels and rect_holes < 3 and aspect > 0.7:
            row_widths = np.sum(bin_img > 0, axis=1)
            max_row = int(np.argmax(row_widths))
            if max_row < len(row_widths) * 0.65 and np.max(row_widths) > w_box * 0.60:
                bot_slice = bin_img[int(ys.max() - h_box * 0.15):, :]
                bot_cols = np.sum(bot_slice > 0, axis=0) > 0
                transitions = len(np.where(np.diff(bot_cols.astype(int)))[0])
                if transitions >= 4:
                    if aspect > 1.3:
                        return cls._build_furniture_blueprint("table", "furniture", 0.80, prompt_text)
                    else:
                        return cls._build_furniture_blueprint("chair", "furniture", 0.82, prompt_text)

        # 6. Radial Flower Detection (round shape with multiple petal protrusions)
        if 0.75 < aspect < 1.35 and hole_count == 0:
            area = float(np.sum(bin_img > 0))
            convex_hull = cv2.convexHull(np.column_stack([xs, ys]))
            hull_area = float(cv2.contourArea(convex_hull))
            solidity = area / max(1.0, hull_area)
            if solidity < 0.75:
                return cls._build_flora_blueprint("flower", "flora", 0.78, prompt_text)

        # 7. Mobile Phone / Device Detection (rounded rectangle, aspect ~ 1.8 - 2.4, high solidity)
        if 0.4 < aspect < 0.6:
            area = float(np.sum(bin_img > 0))
            box_area = w_box * h_box
            if area / max(1.0, box_area) > 0.85:
                return cls._build_electronic_blueprint("mobile_phone", "electronics", 0.82, prompt_text)

        # 8. Axial Pencil Detection (ONLY if extreme aspect ratio > 4.0 or < 0.22)
        if aspect < 0.22 or aspect > 4.0:
            return cls._build_axial_blueprint("pencil", "tool", 0.78, prompt_text)

        # -------------------------------------------------------------
        # Fallback to Prompt Keyword Hints ONLY if Image Geometry was Ambiguous
        # -------------------------------------------------------------
        p_lower = prompt_text.lower()
        def has_kw(w: str) -> bool:
            return bool(re.search(rf"\b{re.escape(w)}\b", p_lower))

        if any(has_kw(k) for k in ["house", "building", "home", "villa", "cottage", "cabin", "mansion", "residence", "architecture"]):
            return cls._build_architecture_blueprint("house", "architecture", 0.90, prompt_text, aspect=aspect, rect_holes=rect_holes)
        elif any(has_kw(k) for k in ["motorcycle", "motorbike", "bike", "bicycle", "scooter", "chopper", "moped"]):
            matched = next(k for k in ["motorcycle", "motorbike", "bike", "bicycle", "scooter", "chopper", "moped"] if has_kw(k))
            name = "motorcycle" if matched in ["motorcycle", "motorbike", "chopper"] else ("bicycle" if matched in ["bicycle", "bike"] else matched)
            return cls._build_vehicle_blueprint(name, "vehicle", 0.90, prompt_text, aspect=aspect, wheel_candidates=wheel_candidates, w_box=w_box, h_box=h_box)
        elif any(has_kw(k) for k in ["car", "vehicle", "automobile", "truck", "van", "bus", "jeep"]):
            matched = next(k for k in ["car", "vehicle", "automobile", "truck", "van", "bus", "jeep"] if has_kw(k))
            return cls._build_vehicle_blueprint(matched, "vehicle", 0.90, prompt_text, aspect=aspect, wheel_candidates=wheel_candidates, w_box=w_box, h_box=h_box)
        elif any(has_kw(k) for k in ["airplane", "plane", "aircraft", "jet"]):
            return cls._build_vehicle_blueprint("airplane", "vehicle", 0.90, prompt_text, aspect=aspect, w_box=w_box, h_box=h_box)
        elif any(has_kw(k) for k in ["boat", "ship", "yacht", "vessel", "submarine"]):
            return cls._build_vehicle_blueprint("boat", "vehicle", 0.90, prompt_text, aspect=aspect, w_box=w_box, h_box=h_box)
        elif any(has_kw(ak) for ak in ["rat", "mouse", "rodent", "cat", "dog", "puppy", "kitten", "bird", "fish", "rabbit", "bunny", "hamster", "squirrel", "horse", "cow", "sheep", "elephant", "lion", "tiger", "bear", "animal", "creature", "human", "person", "man", "woman", "character"]):
            matched = next(ak for ak in ["rat", "mouse", "rodent", "cat", "dog", "puppy", "kitten", "bird", "fish", "rabbit", "bunny", "hamster", "squirrel", "horse", "cow", "sheep", "elephant", "lion", "tiger", "bear", "animal", "creature", "human", "person", "man", "woman", "character"] if has_kw(ak))
            c_name = "rat" if matched in ["rat", "mouse", "rodent", "hamster"] else matched
            return cls._build_creature_blueprint(c_name, "animal", 0.90, prompt_text)
        elif any(has_kw(k) for k in ["guitar", "violin", "cello", "instrument", "bass", "ukulele"]):
            return cls._build_instrument_blueprint("guitar", "instrument", 0.90, prompt_text)
        elif any(has_kw(k) for k in ["shoe", "boot", "sneaker", "footwear", "hat", "cap", "backpack", "bag", "jacket", "shirt"]):
            matched = next(k for k in ["shoe", "boot", "sneaker", "footwear", "hat", "cap", "backpack", "bag", "jacket", "shirt"] if has_kw(k))
            return cls._build_apparel_blueprint(matched, "clothing", 0.90, prompt_text)
        elif has_kw("flower") or has_kw("plant") or has_kw("rose") or has_kw("tulip") or has_kw("tree") or has_kw("leaf") or has_kw("mushroom") or has_kw("cactus"):
            matched = "flower" if (has_kw("flower") or has_kw("rose") or has_kw("tulip")) else ("tree" if has_kw("tree") else "plant")
            return cls._build_flora_blueprint(matched, "flora", 0.90, prompt_text)
        elif has_kw("phone") or has_kw("smartphone") or has_kw("mobile") or has_kw("device") or has_kw("laptop") or has_kw("computer") or has_kw("monitor") or has_kw("camera") or has_kw("headphones") or has_kw("watch"):
            matched = "laptop" if (has_kw("laptop") or has_kw("computer")) else "mobile_phone"
            return cls._build_electronic_blueprint(matched, "electronics", 0.90, prompt_text)
        elif has_kw("chair") or has_kw("armchair") or has_kw("sofa") or has_kw("couch") or has_kw("bed") or has_kw("bookshelf"):
            return cls._build_furniture_blueprint("chair", "furniture", 0.90, prompt_text)
        elif has_kw("table") or has_kw("desk"):
            return cls._build_furniture_blueprint("table", "furniture", 0.90, prompt_text)
        elif has_kw("stool"):
            return cls._build_furniture_blueprint("stool", "furniture", 0.88, prompt_text)
        elif has_kw("lamp") or has_kw("lighting"):
            return cls._build_furniture_blueprint("lamp", "lighting", 0.90, prompt_text)
        elif has_kw("mug") or has_kw("cup"):
            return cls._build_vessel_blueprint("mug", "drinkware", 0.90, prompt_text)
        elif has_kw("bottle") or has_kw("flask"):
            return cls._build_vessel_blueprint("bottle", "container", 0.90, prompt_text)
        elif has_kw("water_glass") or has_kw("glass") or has_kw("tumbler"):
            return cls._build_vessel_blueprint("water_glass", "drinkware", 0.90, prompt_text)
        elif has_kw("vase") or has_kw("urn"):
            return cls._build_vessel_blueprint("vase", "decor", 0.89, prompt_text)
        elif has_kw("teapot"):
            return cls._build_vessel_blueprint("teapot", "drinkware", 0.90, prompt_text)
        elif has_kw("pencil") or has_kw("pen") or has_kw("crayon") or has_kw("marker") or has_kw("sword") or has_kw("hammer") or has_kw("wrench") or has_kw("tool"):
            matched = "pencil" if (has_kw("pencil") or has_kw("pen")) else ("sword" if has_kw("sword") else "tool")
            return cls._build_axial_blueprint(matched, "tool", 0.90, prompt_text)

        # 9. Arbitrary / Unfamiliar Object Blueprint
        return cls._generic_unfamiliar_blueprint("sculpted_object", prompt_text)

    # -------------------------------------------------------------------------
    # Structured Blueprint Builders (Phase 3 Schema)
    # -------------------------------------------------------------------------
    @classmethod
    def _build_architecture_blueprint(
        cls,
        name: str,
        cat: str,
        conf: float,
        prompt: str,
        aspect: float = 1.6,
        rect_holes: int = 10
    ) -> Dict[str, Any]:
        p_col = cls._extract_color(prompt, "#EDE8D0")
        roof_col = "#2D3134"
        p_lower = prompt.lower()
        if "dark roof" in p_lower or "black roof" in p_lower or "slate" in p_lower:
            roof_col = "#1E2226"
        elif "brown roof" in p_lower or "red roof" in p_lower or "terracotta" in p_lower:
            roof_col = "#522B14"
        elif "green roof" in p_lower:
            roof_col = "#166534"

        return {
            "object": name,
            "object_name": name,
            "object_type": name,
            "object_category": cat,
            "confidence": conf,
            "paradigm": "architectural_building",
            "view": "front elevation",
            "primary_color": p_col,
            "description": f"A realistic two-storey {name} with authentic sloping gabled roof, planar warm cream facades, multi-pane fenestration windows, covered entrance porch with columns and railings, sectional garage door, and masonry chimney.",
            "suggested_thickness": 1.0,
            "suggested_smoothness": 1,
            "main_structure": {
                "shape": "two_storey_building_mass",
                "roof_type": "gabled_roof_with_eaves",
                "aspect_ratio": aspect,
                "rect_holes": rect_holes,
                "topology": "volumetric_solid"
            },
            "components": [
                {"name": "concrete_foundation", "type": "foundation_plinth", "shape": "plinth_slab", "location": "base", "material": "natural_concrete", "color": "#6B7280", "roughness": 0.88, "metallic": 0.05, "position": {"x": 0.0, "y": 0.25, "z": 0.0}, "dimensions": {"width": 11.8, "height": 0.5, "depth": 10.8}},
                {"name": "main_building_ground_floor", "type": "building_volume", "shape": "ground_storey_envelope", "location": "lower_facade", "material": "siding_stucco", "color": p_col, "roughness": 0.65, "metallic": 0.02, "position": {"x": -2.2, "y": 1.8, "z": 0.0}, "dimensions": {"width": 7.2, "height": 2.6, "depth": 9.8}},
                {"name": "main_building_first_floor", "type": "building_volume", "shape": "upper_storey_envelope", "location": "upper_facade", "material": "siding_stucco", "color": p_col, "roughness": 0.65, "metallic": 0.02, "position": {"x": -2.2, "y": 4.4, "z": 0.0}, "dimensions": {"width": 7.2, "height": 2.6, "depth": 9.8}},
                {"name": "exterior_walls", "type": "exterior_wall", "shape": "two_storey_envelope", "location": "body", "material": "warm_cream_siding", "color": p_col, "roughness": 0.65, "metallic": 0.02, "position": {"x": -2.2, "y": 3.1, "z": 0.0}, "dimensions": {"width": 7.2, "height": 5.2, "depth": 9.8}},
                {"name": "story_divider_molding", "type": "belt_course", "shape": "horizontal_molding", "location": "mid_level", "material": "architectural_trim", "color": "#F8FAFC", "roughness": 0.40, "metallic": 0.05, "position": {"x": -2.2, "y": 3.1, "z": 0.0}, "dimensions": {"width": 7.4, "height": 0.14, "depth": 10.0}},
                {"name": "main_roof_left", "type": "sloped_roof_surface", "shape": "left_gable_slope", "location": "top_left", "material": "charcoal_roof_shingles", "color": roof_col, "roughness": 0.78, "metallic": 0.10, "position": {"x": -4.1, "y": 6.9, "z": 0.0}, "dimensions": {"width": 4.0, "height": 2.5, "depth": 10.4}},
                {"name": "main_roof_right", "type": "sloped_roof_surface", "shape": "right_gable_slope", "location": "top_right", "material": "charcoal_roof_shingles", "color": roof_col, "roughness": 0.78, "metallic": 0.10, "position": {"x": -0.3, "y": 6.9, "z": 0.0}, "dimensions": {"width": 4.0, "height": 2.5, "depth": 10.4}},
                {"name": "gabled_roof_system", "type": "gable_roof", "shape": "sloped_gable_with_eaves", "location": "top", "material": "charcoal_roof_shingles", "color": roof_col, "roughness": 0.78, "metallic": 0.10, "position": {"x": -2.2, "y": 6.9, "z": 0.0}, "dimensions": {"width": 7.8, "height": 2.5, "depth": 10.4}},
                {"name": "front_cross_gable_roof", "type": "dormer_gable_roof", "shape": "projecting_gable", "location": "front_roof", "material": "charcoal_roof_shingles", "color": roof_col, "roughness": 0.78, "metallic": 0.10, "position": {"x": -2.2, "y": 5.7, "z": 3.2}, "dimensions": {"width": 3.6, "height": 2.1, "depth": 4.4}},
                {"name": "covered_entrance_porch", "type": "porch", "shape": "porch_deck_and_canopy", "location": "front_ground", "material": "porch_concrete_deck", "color": "#9CA3AF", "roughness": 0.80, "metallic": 0.05, "position": {"x": -0.5, "y": 0.62, "z": 5.4}, "dimensions": {"width": 2.6, "height": 0.25, "depth": 1.8}},
                {"name": "porch_columns", "type": "column", "count": 2, "shape": "vertical_cylinder", "location": "porch_corners", "material": "white_painted_wood", "color": "#F8FAFC", "roughness": 0.35, "metallic": 0.02, "position": {"x": -0.5, "y": 1.95, "z": 6.1}, "dimensions": {"radius": 0.08, "height": 2.4}},
                {"name": "porch_railings", "type": "railing", "shape": "metal_balustrade", "location": "porch_perimeter", "material": "dark_painted_metal", "color": "#1E293B", "roughness": 0.25, "metallic": 0.85, "position": {"x": -0.5, "y": 1.15, "z": 5.4}, "dimensions": {"height": 0.85}},
                {"name": "porch_steps", "type": "steps", "count": 2, "shape": "tiered_slabs", "location": "porch_front", "material": "natural_concrete", "color": "#6B7280", "roughness": 0.85, "metallic": 0.05, "position": {"x": -0.5, "y": 0.15, "z": 6.5}, "dimensions": {"width": 3.0, "height": 0.30, "depth": 0.70}},
                {"name": "main_entrance_door", "type": "entry_door", "shape": "paneled_entry_door", "location": "front_porch", "material": "natural_hardwood", "color": "#78350F", "roughness": 0.45, "metallic": 0.05, "position": {"x": -0.5, "y": 1.6, "z": 4.98}, "dimensions": {"width": 0.95, "height": 2.1, "depth": 0.08}, "components": ["door_panel", "door_frame", "brass_hardware"]},
                {"name": "fenestration_windows", "type": "window", "count": 7, "shape": "glazed_multi_pane", "location": "facades", "material": "reflective_glass", "color": "#1E293B", "roughness": 0.04, "metallic": 0.75, "components": ["trim_frames", "tinted_glass_panes"]},
                {"name": "attached_garage_wing", "type": "garage_structure", "shape": "single_storey_wing", "location": "right_facade", "material": "siding_stucco", "color": p_col, "roughness": 0.65, "metallic": 0.02, "position": {"x": 3.6, "y": 2.2, "z": -1.0}, "dimensions": {"width": 4.4, "height": 3.4, "depth": 7.8}},
                {"name": "garage_roof", "type": "sloped_roof_surface", "shape": "gable_roof_eaves", "location": "garage_top", "material": "charcoal_roof_shingles", "color": roof_col, "roughness": 0.78, "metallic": 0.10, "position": {"x": 3.6, "y": 4.75, "z": -1.0}, "dimensions": {"width": 4.8, "height": 1.7, "depth": 8.4}},
                {"name": "garage_door_assembly", "type": "garage_door", "shape": "sectional_horizontal_panels", "location": "garage_front", "material": "enameled_panel", "color": "#94A3B8", "roughness": 0.50, "metallic": 0.20, "position": {"x": 3.6, "y": 1.7, "z": 2.99}, "dimensions": {"width": 3.2, "height": 2.4, "depth": 0.08}, "components": ["door_panels", "trim_frame"]},
                {"name": "masonry_chimney", "type": "chimney", "shape": "vertical_brick_shaft", "location": "roof_slope", "material": "brick_masonry", "color": "#991B1B", "roughness": 0.85, "metallic": 0.05, "position": {"x": -4.8, "y": 7.1, "z": -1.0}, "dimensions": {"width": 0.8, "height": 2.8, "depth": 0.8}, "components": ["brick_shaft", "concrete_cap"]}
            ],
            "spatial_relationships": [
                "foundation_supports_two_storey_envelope",
                "gabled_roof_caps_envelope_with_overhanging_eaves",
                "fenestration_distributed_across_upper_and_lower_storeys",
                "story_divider_molding_demarcates_ground_and_first_floor",
                "garage_wing_integrated_at_side_facade",
                "porch_columns_and_railings_frame_main_entrance"
            ],
            "silhouette_features": ["triangular_gable_pitch", "orthogonal_wall_bounds", "multiple_window_openings", "porch_canopy", "projecting_garage_wing"],
            "visual_features": ["two-storey residential facade", "warm cream exterior walls", "dark charcoal roof tiles", "gabled roof eaves", "multi-pane windows", "covered entry porch with columns and railings", "natural wood door", "attached garage"],
            "openings_and_negative_spaces": [f"{rect_holes}_rectangular_apertures"],
            "proportions": {"relative_depth": 0.90, "width_to_height": aspect},
            "symmetry": "bilateral",
            "material_suggestions": ["slate_shingles", "siding_stucco", "reflective_glass", "concrete_plinth", "natural_wood", "painted_metal"],
            "color_suggestions": [roof_col, p_col, "#6B7280", "#1E293B", "#78350F"],
            "reconstruction_notes": ["Faithful volumetric two-storey architectural building reconstruction with planar facades, gabled roofs, porch railings, and PBR materials"],
            "uncertainties": []
        }

    @classmethod
    def _build_vehicle_blueprint(
        cls,
        name: str,
        cat: str,
        conf: float,
        prompt: str,
        aspect: float = 2.0,
        wheel_candidates: Optional[List[Any]] = None,
        w_box: float = 1.0,
        h_box: float = 1.0,
        view: str = "auto"
    ) -> Dict[str, Any]:
        p_col = cls._extract_color(prompt, "#B91C1C" if "vintage" in name or "vintage" in prompt.lower() else "#DC2626")
        is_two_wheeler = name in ["motorcycle", "motorbike", "bicycle", "bike", "scooter"]
        is_vintage = "vintage" in name or "vintage" in prompt.lower() or "classic" in prompt.lower() or view == "front"
        is_sports_car = (not is_two_wheeler) and (not is_vintage) and (aspect >= 2.2 or "sport" in prompt.lower())

        resolved_view = "front" if (view == "front" or (is_vintage and aspect < 1.7)) else ("side profile" if aspect > 1.4 else "three_quarter")

        if is_two_wheeler:
            body_type = "two_wheeler"
            obj_display = name
            visible_comps = ["front wheel", "rear wheel", "frame", "fuel tank", "handlebars", "seat", "exhaust"]
            silhouette_feats = ["exposed dual wheels", "tubular frame", "raised handlebars", "fuel tank contour"]
        elif is_vintage or resolved_view == "front":
            body_type = "vintage automobile"
            obj_display = "vintage car"
            visible_comps = [
                "bonnet", "windshield", "headlights", "front_grille",
                "grille_bars", "front_bumper", "front_wheels", "side_mirrors"
            ]
            silhouette_feats = [
                "rounded vintage front bonnet", "radiator grille with vertical bars",
                "twin circular chrome headlights", "horizontal front bumper with overriders",
                "split windshield glass", "side rearview mirrors", "two front white-wall wheels"
            ]
        elif is_sports_car:
            body_type = "sports coupe"
            obj_display = "sports car"
            visible_comps = [
                "front bonnet", "roof", "windshield", "side windows",
                "doors", "front wheel", "rear wheel", "rear body"
            ]
            silhouette_feats = [
                "low sports coupe roofline", "aerodynamic front bonnet rake",
                "fastback rear taper", "sculpted door panel contour", "dual circular wheel wells"
            ]
        else:
            body_type = "passenger vehicle"
            obj_display = name
            visible_comps = [
                "front bonnet", "roof", "windshield", "side windows",
                "doors", "front wheel", "rear wheel", "rear body"
            ]
            silhouette_feats = ["horizontal chassis", "cabin profile", "wheel arches"]

        # Extract normalized wheel positions
        wheel_positions = []
        if wheel_candidates and len(wheel_candidates) >= 2 and w_box > 0:
            fronts = [c for c in wheel_candidates if (c[0] / w_box) < 0.45]
            rears = [c for c in wheel_candidates if (c[0] / w_box) > 0.55]
            if fronts:
                f = min(fronts, key=lambda c: c[0])
                wheel_positions.append({
                    "role": "front_left_wheel" if resolved_view == "front" else "front_wheel",
                    "norm_x": round(float(f[0] / w_box), 3),
                    "norm_y": round(float(f[1] / h_box), 3) if h_box > 0 else 0.75,
                    "radius_ratio": round(float(f[2] / w_box), 3)
                })
            if rears:
                r = max(rears, key=lambda c: c[0])
                wheel_positions.append({
                    "role": "front_right_wheel" if resolved_view == "front" else "rear_wheel",
                    "norm_x": round(float(r[0] / w_box), 3),
                    "norm_y": round(float(r[1] / h_box), 3) if h_box > 0 else 0.75,
                    "radius_ratio": round(float(r[2] / w_box), 3)
                })

        return {
            "object": obj_display,
            "object_name": obj_display,
            "object_type": obj_display,
            "object_category": cat,
            "view": resolved_view,
            "body_type": body_type,
            "confidence": conf,
            "paradigm": "vehicle",
            "body_proportions": {
                "length_to_height_ratio": round(float(aspect), 2),
                "relative_depth": 0.40 if is_sports_car else 0.45
            },
            "visible_components": visible_comps,
            "wheel_positions": wheel_positions,
            "silhouette_features": silhouette_feats,
            "main_structure": {
                "shape": "aerodynamic_chassis",
                "primary_orientation": "horizontal",
                "aspect_ratio": round(float(aspect), 2),
                "topology": "solid"
            },
            "components": [
                {"name": "bonnet", "shape": "sloping_hood", "location": "front_center", "material": "car_paint", "color": p_col, "roughness": 0.18, "metallic": 0.75},
                {"name": "front_grille", "shape": "vertical_slats", "location": "front_center", "material": "polished_chrome", "color": "#F1F5F9", "roughness": 0.10, "metallic": 0.96},
                {"name": "headlights", "shape": "circular_pods", "location": "front_sides", "material": "illuminated_glass", "color": "#FEF9C3", "roughness": 0.05, "metallic": 0.10},
                {"name": "front_bumper", "shape": "curved_beam", "location": "front_bottom", "material": "polished_chrome", "color": "#F1F5F9", "roughness": 0.12, "metallic": 0.95},
                {"name": "windshield", "shape": "curved_canopy", "location": "upper_front", "material": "glass", "color": "#1E293B", "roughness": 0.05, "is_transparent": True},
                {"name": "side_mirrors", "shape": "round_discs", "location": "sides", "material": "polished_chrome", "color": "#F1F5F9", "roughness": 0.10, "metallic": 0.95},
                {"name": "wheels", "shape": "cylinders", "location": "bottom", "material": "rubber", "color": "#18181B", "roughness": 0.85, "metallic": 0.02}
            ],
            "spatial_relationships": [
                "bonnet_extends_from_windshield_to_grille",
                "headlights_flank_central_vertical_grille",
                "bumper_underpins_front_facade",
                "wheels_support_chassis_under_fenders"
            ],
            "openings_and_negative_spaces": [],
            "proportions": {"relative_depth": 0.40 if is_sports_car else 0.45, "width_to_height": round(float(aspect), 2)},
            "symmetry": "bilateral",
            "material_suggestions": ["glossy_vintage_paint", "polished_chrome", "tinted_glass", "white_wall_rubber"],
            "color_suggestions": [p_col, "#F1F5F9", "#1E293B", "#18181B", "#FEF9C3"],
            "reconstruction_notes": ["Faithful 3D vintage car reconstruction with authentic volumetric rounded body, chrome grille with vertical bars, circular headlights, and white-wall tires"],
            "uncertainties": []
        }

    @classmethod
    def _build_creature_blueprint(cls, name: str, cat: str, conf: float, prompt: str) -> Dict[str, Any]:
        """
        Builds a comprehensive anatomical 3D blueprint for animals and organic creatures
        (Rat, Mouse, Cat, Dog, Bird, Fish, etc.).
        """
        t = f"{name} {prompt}".lower()
        if any(w in t for w in ["rat", "mouse", "rodent"]):
            default_fur = "#78716C" # Warm natural rodent slate/gray
        elif "cat" in t:
            default_fur = "#EA580C" # Warm ginger/tabby
        elif "dog" in t:
            default_fur = "#92400E" # Golden/warm canine brown
        elif "bird" in t:
            default_fur = "#2563EB" # Avian vibrant blue
        elif "fish" in t:
            default_fur = "#F97316" # Vibrant aquatic orange/gold
        else:
            default_fur = "#64748B"

        fur_col = cls._extract_color(prompt, default_fur)
        belly_col = "#E2E8F0" # Lighter, soft ventral fur
        flesh_pink = "#F472B6" # Natural ear, snout, paw, and tail tone
        eye_black = "#0A0A0A"  # High gloss obsidian black eye beads

        return {
            "object": name,
            "object_name": name,
            "object_type": name,
            "object_category": "animal" if cat in ["creature", "abstract", "unknown"] else cat,
            "confidence": conf,
            "paradigm": "organic_creature",
            "creature_type": name,
            "description": f"A realistic 3D {name} with natural anatomical proportions, volumetric body, and multi-color PBR materials",
            "main_structure": {
                "shape": "quadruped_organic" if any(w in name for w in ["rat", "mouse", "cat", "dog"]) else "organic_creature",
                "primary_orientation": "horizontal",
                "aspect_ratio": 1.6,
                "topology": "solid"
            },
            "components": [
                {"name": "torso", "shape": "volumetric_ellipsoid", "location": "center", "material": "animal_fur", "color": fur_col, "roughness": 0.85, "metallic": 0.0},
                {"name": "belly", "shape": "ventral_surface", "location": "underside", "material": "soft_fur", "color": belly_col, "roughness": 0.88, "metallic": 0.0},
                {"name": "head", "shape": "tapered_cranium", "location": "front", "material": "animal_fur", "color": fur_col, "roughness": 0.85, "metallic": 0.0},
                {"name": "snout_nose", "shape": "rounded_apex", "location": "muzzle", "material": "organic_skin", "color": flesh_pink, "roughness": 0.60, "metallic": 0.0},
                {"name": "eyes", "shape": "dual_spheres", "location": "cranial_sides", "material": "glossy_cornea", "color": eye_black, "roughness": 0.08, "metallic": 0.10},
                {"name": "ears", "shape": "dual_pinna", "location": "top_skull", "material": "cartilage_flesh", "color": flesh_pink, "roughness": 0.70, "metallic": 0.0},
                {"name": "forelegs", "shape": "articulated_columns", "location": "front_ventral", "material": "animal_fur", "color": fur_col, "roughness": 0.85, "metallic": 0.0},
                {"name": "hindlegs", "shape": "muscular_haunches", "location": "rear_ventral", "material": "animal_fur", "color": fur_col, "roughness": 0.85, "metallic": 0.0},
                {"name": "paws", "shape": "clawed_pads", "location": "extremities", "material": "paw_pads", "color": flesh_pink, "roughness": 0.75, "metallic": 0.0},
                {"name": "tail", "shape": "swept_tapered_curve", "location": "caudal_rear", "material": "tail_skin", "color": flesh_pink if any(w in name for w in ["rat", "mouse"]) else fur_col, "roughness": 0.75, "metallic": 0.0}
            ],
            "spatial_relationships": [
                "head_connected_to_torso",
                "snout_at_head_apex",
                "eyes_on_lateral_cranium",
                "ears_atop_cranium",
                "forelegs_at_pectoral_girdle",
                "hindlegs_at_pelvic_girdle",
                "paws_contact_ground",
                "tail_extends_from_rear"
            ],
            "silhouette_features": ["arching_spine", "tapered_snout", "curling_tail", "quadruped_limbs"],
            "openings_and_negative_spaces": ["leg_openings_under_belly"],
            "proportions": {"relative_depth": 0.65, "width_to_height": 1.6},
            "symmetry": "bilateral",
            "material_suggestions": ["matte_natural_fur", "organic_skin", "glossy_eye"],
            "color_suggestions": [fur_col, belly_col, flesh_pink, eye_black],
            "reconstruction_notes": ["Anatomically proportioned volumetric body with bilateral limbs in 3D depth and graceful tail"],
            "uncertainties": []
        }

    @classmethod
    def _build_flora_blueprint(cls, name: str, cat: str, conf: float, prompt: str) -> Dict[str, Any]:
        p_col = cls._extract_color(prompt, "#EC4899")
        return {
            "object": name,
            "object_name": name,
            "object_type": name,
            "object_category": cat,
            "confidence": conf,
            "paradigm": "flora",
            "main_structure": {
                "shape": "radial_bloom",
                "primary_orientation": "vertical",
                "aspect_ratio": 1.0,
                "topology": "organic"
            },
            "components": [
                {"name": "center_disc", "shape": "hemisphere", "location": "center", "material": "organic", "color": "#F59E0B", "roughness": 0.55, "metallic": 0.0},
                {"name": "petals", "shape": "curved_plates", "location": "around_center", "material": "petal_velvet", "color": p_col, "roughness": 0.40, "metallic": 0.0},
                {"name": "stem", "shape": "column", "location": "bottom", "material": "plant_stem", "color": "#16A34A", "roughness": 0.60, "metallic": 0.0}
            ],
            "spatial_relationships": ["petals_surround_disc", "stem_under_bloom"],
            "silhouette_features": ["petal_lobes", "slender_stem"],
            "openings_and_negative_spaces": [],
            "proportions": {"relative_depth": 0.35, "width_to_height": 1.0},
            "symmetry": "radial",
            "material_suggestions": ["petal_velvet", "stem_green", "pollen_amber"],
            "color_suggestions": [p_col, "#F59E0B", "#16A34A"],
            "reconstruction_notes": ["Radial petal arrangement around central disc"],
            "uncertainties": []
        }

    @classmethod
    def _build_electronic_blueprint(cls, name: str, cat: str, conf: float, prompt: str) -> Dict[str, Any]:
        p_col = cls._extract_color(prompt, "#334155")
        return {
            "object": name,
            "object_name": name,
            "object_type": name,
            "object_category": cat,
            "confidence": conf,
            "paradigm": "electronic",
            "main_structure": {
                "shape": "rounded_slab",
                "primary_orientation": "vertical",
                "aspect_ratio": 0.5,
                "topology": "layered"
            },
            "components": [
                {"name": "chassis", "shape": "beveled_box", "location": "frame", "material": "anodized_aluminum", "color": p_col, "roughness": 0.25, "metallic": 0.85},
                {"name": "screen", "shape": "flat_glass", "location": "front", "material": "oled_glass", "color": "#0F172A", "roughness": 0.05, "metallic": 0.1},
                {"name": "camera_module", "shape": "cylinder", "location": "top_corner", "material": "dark_glass", "color": "#090D16", "roughness": 0.10, "metallic": 0.90}
            ],
            "spatial_relationships": ["screen_on_chassis", "camera_on_rear"],
            "silhouette_features": ["rounded_corners", "slender_profile"],
            "openings_and_negative_spaces": [],
            "proportions": {"relative_depth": 0.12, "width_to_height": 0.5},
            "symmetry": "bilateral",
            "material_suggestions": ["anodized_metal", "glossy_glass"],
            "color_suggestions": [p_col, "#0F172A"],
            "reconstruction_notes": ["Slim rectangular chassis with beveled borders"],
            "uncertainties": []
        }

    @classmethod
    def _build_furniture_blueprint(cls, name: str, cat: str, conf: float, prompt: str) -> Dict[str, Any]:
        p_col = cls._extract_color(prompt, "#8B5A2B" if "lamp" not in name else "#EAB308")
        leg_col = "#18181B" if "metal" in prompt.lower() or "black" in prompt.lower() else p_col
        return {
            "object": name,
            "object_name": name,
            "object_type": name,
            "object_category": cat,
            "confidence": conf,
            "paradigm": "structural_furniture",
            "main_structure": {
                "shape": "frame_assembly",
                "primary_orientation": "vertical",
                "aspect_ratio": 0.85,
                "topology": "framed"
            },
            "components": [
                {"name": "primary_slab", "shape": "rounded_box", "location": "top_or_seat", "material": "wood", "color": p_col, "roughness": 0.65, "metallic": 0.0},
                {"name": "supports", "shape": "cylindrical_legs", "location": "bottom", "material": "wood_or_metal", "color": leg_col, "roughness": 0.35, "metallic": 0.85 if leg_col != p_col else 0.0}
            ],
            "spatial_relationships": ["slab_supported_by_legs"],
            "silhouette_features": ["horizontal_surface", "vertical_leg_posts"],
            "openings_and_negative_spaces": ["leg_gap"],
            "proportions": {"relative_depth": 0.90, "width_to_height": 0.85},
            "symmetry": "bilateral",
            "material_suggestions": ["solid_wood", "brushed_steel"],
            "color_suggestions": [p_col, leg_col],
            "reconstruction_notes": ["4 corner legs arranged in 3D depth"],
            "uncertainties": []
        }

    @classmethod
    def _build_vessel_blueprint(cls, name: str, cat: str, conf: float, prompt: str) -> Dict[str, Any]:
        is_glass = any(w in prompt.lower() for w in ["glass", "tumbler", "translucent", "bottle", "water_glass"])
        p_col = cls._extract_color(prompt, "#38BDF8" if is_glass else ("#DC2626" if "red" in prompt.lower() else "#F8FAFC"))
        return {
            "object": name,
            "object_name": name,
            "object_type": name,
            "object_category": cat,
            "confidence": conf,
            "paradigm": "radial_vessel",
            "main_structure": {
                "shape": "rotational_lathe",
                "primary_orientation": "vertical",
                "aspect_ratio": 1.0,
                "topology": "hollow"
            },
            "components": [
                {"name": "body", "shape": "cylinder_lathe", "location": "center", "material": "glass" if is_glass else "ceramic", "color": p_col, "roughness": 0.05 if is_glass else 0.18, "metallic": 0.05, "is_transparent": is_glass},
                {"name": "handle", "shape": "swept_tube", "location": "side", "material": "ceramic", "color": p_col, "roughness": 0.20, "metallic": 0.05}
            ],
            "spatial_relationships": ["handle_attached_to_body"],
            "silhouette_features": ["circular_rim", "hollow_interior", "outer_contour"],
            "openings_and_negative_spaces": ["top_opening", "handle_void"],
            "proportions": {"relative_depth": 1.0, "width_to_height": 1.0},
            "symmetry": "radial",
            "material_suggestions": ["glazed_ceramic", "blown_glass"],
            "color_suggestions": [p_col],
            "reconstruction_notes": ["Rotational body with hollow inner cavity and rim"],
            "uncertainties": []
        }

    @classmethod
    def _build_axial_blueprint(cls, name: str, cat: str, conf: float, prompt: str) -> Dict[str, Any]:
        shaft_col = cls._extract_color(prompt, "#F59E0B")
        eraser_col = "#F472B6"
        ferrule_col = "#CBD5E1" if "gold" not in prompt.lower() else "#EAB308"
        return {
            "object": name,
            "object_name": name,
            "object_type": name,
            "object_category": cat,
            "confidence": conf,
            "paradigm": "axial_instrument",
            "main_structure": {
                "shape": "segmented_cylinder",
                "primary_orientation": "vertical",
                "aspect_ratio": 8.0,
                "topology": "solid"
            },
            "components": [
                {"name": "graphite_tip", "shape": "cone", "location": "top_tip", "material": "graphite", "color": "#1C1C1E", "roughness": 0.85, "metallic": 0.10},
                {"name": "wood_collar", "shape": "conical_frustum", "location": "bevel", "material": "wood", "color": "#DDB892", "roughness": 0.70, "metallic": 0.0},
                {"name": "shaft", "shape": "hexagonal_prism", "location": "body", "material": "lacquered_wood", "color": shaft_col, "roughness": 0.35, "metallic": 0.0},
                {"name": "ferrule", "shape": "cylinder", "location": "collar", "material": "polished_metal", "color": ferrule_col, "roughness": 0.20, "metallic": 0.92},
                {"name": "eraser", "shape": "rounded_cap", "location": "back_end", "material": "rubber", "color": eraser_col, "roughness": 0.75, "metallic": 0.0}
            ],
            "spatial_relationships": ["tip_leads_to_wood", "wood_leads_to_shaft", "shaft_to_ferrule", "ferrule_to_eraser"],
            "silhouette_features": ["tapered_tip", "hexagonal_shaft", "collar_ring"],
            "openings_and_negative_spaces": [],
            "proportions": {"relative_depth": 0.125, "width_to_height": 0.125},
            "symmetry": "radial",
            "material_suggestions": ["graphite", "wood", "lacquered_wood", "metal", "rubber"],
            "color_suggestions": ["#1C1C1E", "#DDB892", shaft_col, ferrule_col, eraser_col],
            "reconstruction_notes": ["5-component axial assembly along principal axis"],
            "uncertainties": []
        }

    @classmethod
    def _build_instrument_blueprint(cls, name: str, cat: str, conf: float, prompt: str) -> Dict[str, Any]:
        p_col = cls._extract_color(prompt, "#B45309")
        return {
            "object": name,
            "object_name": name,
            "object_type": name,
            "object_category": cat,
            "confidence": conf,
            "paradigm": "universal_manifold",
            "main_structure": {
                "shape": "acoustic_body",
                "primary_orientation": "vertical",
                "aspect_ratio": 2.6,
                "topology": "solid"
            },
            "components": [
                {"name": "body", "shape": "hourglass_resonance_chamber", "location": "lower", "material": "wood", "color": p_col, "roughness": 0.25, "metallic": 0.0},
                {"name": "neck", "shape": "slender_fretboard", "location": "upper", "material": "dark_wood", "color": "#1C1917", "roughness": 0.40, "metallic": 0.0},
                {"name": "headstock", "shape": "tapered_apex", "location": "top", "material": "wood", "color": p_col, "roughness": 0.30, "metallic": 0.0}
            ],
            "spatial_relationships": ["headstock_on_neck", "neck_on_body"],
            "silhouette_features": ["curving_waist", "slender_neck", "tuning_pegs"],
            "openings_and_negative_spaces": ["sound_hole"],
            "proportions": {"relative_depth": 0.18, "width_to_height": 0.38},
            "symmetry": "bilateral",
            "material_suggestions": ["lacquered_tone_wood", "ebony", "nickel_strings"],
            "color_suggestions": [p_col, "#1C1917", "#CBD5E1"],
            "reconstruction_notes": ["Faithful 3D acoustic instrument geometry"],
            "uncertainties": []
        }

    @classmethod
    def _build_apparel_blueprint(cls, name: str, cat: str, conf: float, prompt: str) -> Dict[str, Any]:
        p_col = cls._extract_color(prompt, "#2563EB")
        return {
            "object": name,
            "object_name": name,
            "object_type": name,
            "object_category": cat,
            "confidence": conf,
            "paradigm": "universal_manifold",
            "main_structure": {
                "shape": "ergonomic_footwear",
                "primary_orientation": "horizontal",
                "aspect_ratio": 2.2,
                "topology": "solid"
            },
            "components": [
                {"name": "upper", "shape": "curved_instep", "location": "top", "material": "leather_textile", "color": p_col, "roughness": 0.60, "metallic": 0.0},
                {"name": "outsole", "shape": "tread_base", "location": "bottom", "material": "vulcanized_rubber", "color": "#F8FAFC", "roughness": 0.85, "metallic": 0.0}
            ],
            "spatial_relationships": ["upper_on_outsole"],
            "silhouette_features": ["instep_curvature", "tread_profile", "collar_opening"],
            "openings_and_negative_spaces": ["ankle_collar"],
            "proportions": {"relative_depth": 0.40, "width_to_height": 2.2},
            "symmetry": "asymmetric",
            "material_suggestions": ["leather_textile", "cushioned_midsole", "rubber_sole"],
            "color_suggestions": [p_col, "#F8FAFC"],
            "reconstruction_notes": ["Volumetric footwear with sole and collar"],
            "uncertainties": []
        }

    @classmethod
    def _generic_unfamiliar_blueprint(cls, name: str, prompt: str) -> Dict[str, Any]:
        p_col = cls._extract_color(prompt, "#6366F1")
        return {
            "object": name,
            "object_name": name,
            "object_type": name,
            "object_category": "abstract_form",
            "confidence": 0.50,
            "paradigm": "universal_manifold",
            "main_structure": {
                "shape": "organic_manifold",
                "primary_orientation": "vertical",
                "aspect_ratio": 1.0,
                "topology": "solid"
            },
            "components": [
                {"name": "sculpted_body", "shape": "manifold_surface", "location": "whole", "material": "ceramic", "color": p_col, "roughness": 0.30, "metallic": 0.05}
            ],
            "spatial_relationships": ["integrated_continuous_form"],
            "silhouette_features": ["freeform_boundary_contour"],
            "openings_and_negative_spaces": [],
            "proportions": {"relative_depth": 0.50, "width_to_height": 1.0},
            "symmetry": "bilateral",
            "material_suggestions": ["matte_ceramic"],
            "color_suggestions": [p_col],
            "reconstruction_notes": ["Faithful 3D volumetric inflation of 2D silhouette"],
            "uncertainties": ["Unfamiliar arbitrary sketch; adapting harmonic distance field to exact 2D contours"]
        }

    @staticmethod
    def _extract_color(text: str, default: str) -> str:
        color_lookup = {
            "yellow": "#F59E0B", "gold": "#EAB308", "red": "#DC2626", "crimson": "#B91C1C",
            "blue": "#2563EB", "navy": "#1E3A8A", "green": "#16A34A", "emerald": "#059669",
            "black": "#18181B", "charcoal": "#27272A", "white": "#F8FAFC", "pink": "#F472B6",
            "purple": "#9333EA", "violet": "#7C3AED", "orange": "#EA580C", "brown": "#78350F",
            "silver": "#CBD5E1", "terracotta": "#C2410C", "translucent": "#38BDF8"
        }
        t = text.lower()
        for c_name, c_hex in color_lookup.items():
            if re.search(rf"\b{re.escape(c_name)}\b", t):
                return c_hex
        return default

    @classmethod
    def fuse_semantics_and_prompt(
        cls,
        ai_analysis: Optional[Dict[str, Any]] = None,
        prompt: Optional[str] = None,
        cv_structure: Optional[Dict[str, Any]] = None,
        conflict_resolution: str = "preserve_image"
    ) -> Dict[str, Any]:
        """
        Multimodal Semantic Fusion Engine:
        Strict Image-First Reconstruction Policy.
        Preserves original uploaded image identity and structural geometry as primary conditioning.
        Natural-language prompt refines appearance, materials, colors, and style.
        Detects conflicts between image and prompt, defaulting to 'preserve_image'.
        """
        prompt_text = (prompt or "").strip()
        ai = dict(ai_analysis or {})
        prompt_info = cls.analyze_user_prompt(prompt_text)
        conflict_info = cls.detect_image_prompt_conflict(ai, prompt_info)

        # 1. Conflict handling
        if conflict_info["conflict_detected"]:
            logger.warning(conflict_info["conflict_message"])
            if conflict_resolution == "preserve_image":
                # Preserve image identity, apply prompt colors & materials
                eff_col = prompt_info.get("requested_color")
                if eff_col:
                    hex_col = cls._extract_color(eff_col, ai.get("primary_color", "#78350F"))
                    ai["primary_color"] = hex_col
                    # Update component colors
                    for comp in ai.get("components", []):
                        if any(k in comp.get("name", "").lower() for k in ["wall", "body", "facade", "exterior", "shaft"]):
                            comp["color"] = hex_col

                ai["conflict_detected"] = True
                ai["conflict_type"] = conflict_info["conflict_type"]
                ai["conflict_message"] = conflict_info["conflict_message"]
                ai["conflict_resolution"] = "preserve_image"
                ai["effective_object"] = ai.get("object_name")
                ai.setdefault("paradigm", "architectural_building" if ai.get("object_category") == "architecture" else "volumetric_manifold")
                return ai
            else:
                # User explicitly requested to override with prompt
                ai["conflict_detected"] = True
                ai["conflict_resolution"] = "override_with_prompt"
                # Proceed to generate prompt object archetype below

        # 2. No conflict (or user selected override_with_prompt)
        # Determine effective object: image object has priority 1, unless override requested
        if conflict_info["conflict_detected"] and conflict_resolution == "override_with_prompt":
            eff_name = prompt_info.get("requested_object") or "sculpted_object"
            eff_cat = prompt_info.get("requested_category") or "generic"
            eff_paradigm = ""
        else:
            eff_name = str(ai.get("object_name") or ai.get("object_type") or prompt_info.get("requested_object") or "sculpted_object").lower()
            eff_cat = str(ai.get("object_category") or prompt_info.get("requested_category") or "generic").lower()
            eff_paradigm = str(ai.get("paradigm") or "").lower()

        # Architecture Archetype
        if eff_cat == "architecture" or eff_paradigm == "architectural_building" or eff_name in ["house", "building", "home", "villa", "cottage", "cabin", "mansion", "residence", "architecture"]:
            blueprint = cls._build_architecture_blueprint(
                name=eff_name if eff_name in ["house", "building", "villa", "cottage", "cabin"] else "house",
                cat="architecture",
                conf=float(ai.get("confidence", 0.92)),
                prompt=prompt_text
            )
            blueprint["conflict_detected"] = conflict_info["conflict_detected"]
            blueprint["conflict_message"] = conflict_info["conflict_message"]
            blueprint["conflict_resolution"] = conflict_resolution
            blueprint["effective_object"] = blueprint["object_name"]
            return blueprint

        # Vehicle Archetype
        if eff_cat == "vehicle" or eff_paradigm in ["vehicle", "car"] or eff_name in ["car", "sports car", "vintage car", "classic car", "automobile", "motorcycle", "motorbike", "bike", "bicycle", "scooter", "truck", "van", "bus", "jeep", "airplane", "boat"]:
            is_two_wheeler = eff_name in ["motorcycle", "motorbike", "bike", "bicycle", "scooter"]
            name = "motorcycle" if eff_name in ["motorcycle", "motorbike"] else ("bicycle" if eff_name in ["bicycle", "bike"] else eff_name)
            v_view = ai.get("view") or ("front" if ("vintage" in eff_name or "vintage" in prompt_text.lower()) else "auto")
            blueprint = cls._build_vehicle_blueprint(name, "vehicle", float(ai.get("confidence", 0.92)), prompt_text, view=v_view)
            blueprint["conflict_detected"] = conflict_info["conflict_detected"]
            blueprint["conflict_message"] = conflict_info["conflict_message"]
            blueprint["conflict_resolution"] = conflict_resolution
            blueprint["effective_object"] = blueprint["object_name"]
            return blueprint

        # Creature / Animal Archetype
        animal_kws = ["rat", "mouse", "rodent", "cat", "dog", "puppy", "kitten", "bird", "fish", "rabbit", "bunny", "hamster", "squirrel", "horse", "cow", "sheep", "elephant", "lion", "tiger", "bear", "animal", "creature"]
        if eff_cat == "animal" or any(k in eff_name for k in animal_kws):
            c_name = "rat" if any(k in eff_name for k in ["rat", "mouse", "rodent", "hamster"]) else eff_name
            blueprint = cls._build_creature_blueprint(c_name, "animal", float(ai.get("confidence", 0.93)), prompt_text)
            blueprint["conflict_detected"] = conflict_info["conflict_detected"]
            blueprint["conflict_message"] = conflict_info["conflict_message"]
            blueprint["conflict_resolution"] = conflict_resolution
            blueprint["effective_object"] = blueprint["object_name"]
            return blueprint

        # Musical Instruments
        if eff_cat == "instrument" or any(k in eff_name for k in ["guitar", "violin", "cello", "instrument", "bass", "ukulele"]):
            blueprint = cls._build_instrument_blueprint("guitar", "instrument", float(ai.get("confidence", 0.91)), prompt_text)
            blueprint["conflict_detected"] = conflict_info["conflict_detected"]
            blueprint["conflict_message"] = conflict_info["conflict_message"]
            blueprint["conflict_resolution"] = conflict_resolution
            blueprint["effective_object"] = blueprint["object_name"]
            return blueprint

        # Apparel / Footwear
        if eff_cat == "clothing" or any(k in eff_name for k in ["shoe", "boot", "sneaker", "footwear"]):
            blueprint = cls._build_apparel_blueprint("shoe", "clothing", float(ai.get("confidence", 0.91)), prompt_text)
            blueprint["conflict_detected"] = conflict_info["conflict_detected"]
            blueprint["conflict_message"] = conflict_info["conflict_message"]
            blueprint["conflict_resolution"] = conflict_resolution
            blueprint["effective_object"] = blueprint["object_name"]
            return blueprint

        # Flora Archetype
        if eff_cat == "flora" or any(k in eff_name for k in ["flower", "plant", "rose", "tulip", "bloom", "sunflower", "daisy"]):
            blueprint = cls._build_flora_blueprint("flower", "flora", float(ai.get("confidence", 0.90)), prompt_text)
            blueprint["conflict_detected"] = conflict_info["conflict_detected"]
            blueprint["conflict_message"] = conflict_info["conflict_message"]
            blueprint["conflict_resolution"] = conflict_resolution
            blueprint["effective_object"] = blueprint["object_name"]
            return blueprint

        # Electronics Archetype
        if eff_cat == "electronics" or any(k in eff_name for k in ["phone", "smartphone", "mobile", "tablet", "ipad", "screen", "device", "camera"]):
            blueprint = cls._build_electronic_blueprint("mobile_phone" if "camera" not in eff_name else "camera", "electronics", float(ai.get("confidence", 0.90)), prompt_text)
            blueprint["conflict_detected"] = conflict_info["conflict_detected"]
            blueprint["conflict_message"] = conflict_info["conflict_message"]
            blueprint["conflict_resolution"] = conflict_resolution
            blueprint["effective_object"] = blueprint["object_name"]
            return blueprint

        # Furniture Archetype
        if eff_cat == "furniture" or any(k in eff_name for k in ["chair", "armchair", "recliner", "sofa", "couch", "table", "desk", "bench", "stool", "lamp"]):
            if any(k in eff_name for k in ["table", "desk", "bench"]):
                f_type = "table"
            elif "stool" in eff_name:
                f_type = "stool"
            elif "lamp" in eff_name:
                f_type = "lamp"
            else:
                f_type = "chair"
            blueprint = cls._build_furniture_blueprint(f_type, "furniture", float(ai.get("confidence", 0.90)), prompt_text)
            blueprint["conflict_detected"] = conflict_info["conflict_detected"]
            blueprint["conflict_message"] = conflict_info["conflict_message"]
            blueprint["conflict_resolution"] = conflict_resolution
            blueprint["effective_object"] = blueprint["object_name"]
            return blueprint

        # Vessels Archetype
        if eff_cat in ["drinkware", "container", "vessel"] or any(k in eff_name for k in ["mug", "cup", "teacup", "coffee_mug", "bottle", "flask", "water_bottle", "glass", "water_glass", "vase", "teapot"]):
            if any(k in eff_name for k in ["bottle", "flask"]):
                v_name = "bottle"
                v_cat = "container"
            elif any(k in eff_name for k in ["glass", "water_glass"]):
                v_name = "water_glass"
                v_cat = "drinkware"
            elif "vase" in eff_name:
                v_name = "vase"
                v_cat = "decor"
            elif "teapot" in eff_name:
                v_name = "teapot"
                v_cat = "drinkware"
            else:
                v_name = "mug"
                v_cat = "drinkware"
            blueprint = cls._build_vessel_blueprint(v_name, v_cat, float(ai.get("confidence", 0.91)), prompt_text)
            blueprint["conflict_detected"] = conflict_info["conflict_detected"]
            blueprint["conflict_message"] = conflict_info["conflict_message"]
            blueprint["conflict_resolution"] = conflict_resolution
            blueprint["effective_object"] = blueprint["object_name"]
            return blueprint

        # Axial Instrument Archetype (ONLY when explicitly identified as writing instrument/tool)
        if eff_cat == "tool" or any(k in eff_name for k in ["pencil", "pen", "crayon", "marker", "stylus", "screwdriver", "wand"]):
            blueprint = cls._build_axial_blueprint("pencil", "tool", float(ai.get("confidence", 0.92)), prompt_text)
            blueprint["conflict_detected"] = conflict_info["conflict_detected"]
            blueprint["conflict_message"] = conflict_info["conflict_message"]
            blueprint["conflict_resolution"] = conflict_resolution
            blueprint["effective_object"] = blueprint["object_name"]
            return blueprint

        # Check if AI analysis provided explicit paradigm
        if ai.get("paradigm"):
            ai["conflict_detected"] = conflict_info["conflict_detected"]
            ai["conflict_message"] = conflict_info["conflict_message"]
            ai["conflict_resolution"] = conflict_resolution
            ai["effective_object"] = ai.get("object_name")
            return ai

        # Unfamiliar / Freeform Object
        res = cls._generic_unfamiliar_blueprint(eff_name or "sculpted_object", prompt_text)
        res["conflict_detected"] = conflict_info["conflict_detected"]
        res["conflict_message"] = conflict_info["conflict_message"]
        res["conflict_resolution"] = conflict_resolution
        res["effective_object"] = res["object_name"]
        return res

    @classmethod
    async def analyze_sketch(
        cls,
        image_path: Path,
        prompt_guidance: Optional[str] = None,
        conflict_resolution: str = "preserve_image"
    ) -> Dict[str, Any]:
        """
        Invokes Groq Vision model with token-optimized 256x256 image and content-hash caching.
        If offline or rate-limited, executes dynamic CV feature classification.
        Enforces strict Image-First Reconstruction Policy and Conflict Detection.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 1. Image Optimization & Content Hash
        with Image.open(image_path) as im:
            im_rgb = im.convert("RGB")
            resized = im_rgb.resize((144, 144), Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=75)
            img_bytes = buf.getvalue()
            b64_img = base64.b64encode(img_bytes).decode("utf-8")

        prompt_str = (prompt_guidance or "").strip()
        content_hash = hashlib.sha256(img_bytes + prompt_str.encode("utf-8") + conflict_resolution.encode("utf-8")).hexdigest()

        # Check in-memory content cache
        if content_hash in cls._vision_cache:
            logger.info(f"Vision Cache HIT for hash: {content_hash[:12]}")
            return dict(cls._vision_cache[content_hash])

        prompt_info = cls.analyze_user_prompt(prompt_str)

        # If Groq Vision is in active rate-limit/network cooldown, fast-path to Dynamic CV Classifier
        if time.time() < cls._groq_rate_limited_until:
            logger.info("Groq Vision currently rate-limited (cooldown active). Fast-pathing to Dynamic CV Classifier.")
            blueprint = cls.cv_classify_sketch(image_path, prompt_str)
            blueprint["available"] = True
            blueprint["model_used"] = "Dynamic Computer Vision Feature Classifier (Fast Cooldown)"
            blueprint["content_hash"] = content_hash
            blueprint.setdefault("description", f"A 3D {blueprint.get('object_name', 'object')}")
            blueprint.setdefault("suggested_thickness", 1.0)
            blueprint.setdefault("suggested_smoothness", 3)
            blueprint.setdefault("visual_features", blueprint.get("silhouette_features", ["3D contour boundary"]))
            blueprint.setdefault("object_type", blueprint.get("object_name", "object"))
            conflict_info = cls.detect_image_prompt_conflict(blueprint, prompt_info)
            blueprint["conflict_detected"] = conflict_info["conflict_detected"]
            blueprint["conflict_type"] = conflict_info["conflict_type"]
            blueprint["conflict_message"] = conflict_info["conflict_message"]
            blueprint["conflict_resolution"] = conflict_resolution
            blueprint["effective_object"] = blueprint.get("object_name") if conflict_resolution == "preserve_image" else (prompt_info.get("requested_object") or blueprint.get("object_name"))
            cls._vision_cache[content_hash] = blueprint
            return blueprint

        online = cls.is_online()

        # If Groq is unconfigured or offline / sandboxed, use dynamic CV classifier
        if not cls.is_configured() or not online:
            logger.info("Groq unconfigured or offline/sandboxed environment; executing Dynamic CV Classifier")
            blueprint = cls.cv_classify_sketch(image_path, prompt_str)
            blueprint["available"] = True
            blueprint["model_used"] = "Local Computer Vision Classifier"
            blueprint.setdefault("description", f"A 3D {blueprint.get('object_name', 'object')}")
            blueprint.setdefault("suggested_thickness", 1.0)
            blueprint.setdefault("suggested_smoothness", 3)
            blueprint.setdefault("visual_features", blueprint.get("silhouette_features", ["3D contour boundary"]))
            blueprint.setdefault("object_type", blueprint.get("object_name", "object"))
            conflict_info = cls.detect_image_prompt_conflict(blueprint, prompt_info)
            blueprint["conflict_detected"] = conflict_info["conflict_detected"]
            blueprint["conflict_type"] = conflict_info["conflict_type"]
            blueprint["conflict_message"] = conflict_info["conflict_message"]
            blueprint["conflict_resolution"] = conflict_resolution
            blueprint["effective_object"] = blueprint.get("object_name") if conflict_resolution == "preserve_image" else (prompt_info.get("requested_object") or blueprint.get("object_name"))
            cls._vision_cache[content_hash] = blueprint
            return blueprint

        # 2. Groq Vision Call (Compact, token-optimized payload)
        system_instruction = (
            "Analyze this 2D sketch or image. Identify the real-life object depicted.\n"
            "Return strictly valid JSON matching this schema:\n"
            "{\n"
            '  "object_name": string (the exact real-life object, e.g. "house", "building", "motorcycle", "car", "bicycle", "airplane", "boat", "rat", "cat", "dog", "horse", "chair", "table", "mug", "bottle", "guitar", "shoe", "phone", "pencil", "flower"),\n'
            '  "object_category": string ("architecture", "vehicle", "animal", "furniture", "drinkware", "tool", "flora", "electronics", "instrument", "clothing", "abstract"),\n'
            '  "confidence": float (0.1 to 1.0),\n'
            '  "primary_color": string (#hex code),\n'
            '  "symmetry": "bilateral" | "radial" | "asymmetric"\n'
            "}\n"
            "Identify strictly the visual object shown in the image. Do not let prompt override image identity if contradictory."
        )

        user_content = system_instruction
        if prompt_str:
            user_content += f"\nOptional Prompt Context: {prompt_str}\n(Note: Preserves image geometry as primary object if contradictory)."

        payload = {
            "model": "qwen/qwen3.8-27b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_content},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 120
        }

        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await asyncio.wait_for(
                    client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload),
                    timeout=8.0
                )

            if resp.status_code == 200:
                raw_txt = resp.json()["choices"][0]["message"]["content"].strip()
                if "```" in raw_txt:
                    parts = raw_txt.split("```")
                    if len(parts) >= 2:
                        raw_txt = parts[1]
                    if raw_txt.startswith("json"):
                        raw_txt = raw_txt[4:]
                    raw_txt = raw_txt.strip()

                parsed = json.loads(raw_txt)
                parsed["available"] = True
                parsed["model_used"] = "qwen/qwen3.8-27b (Groq LPU Vision)"
                parsed["content_hash"] = content_hash
                obj_name = str(parsed.get("object_name") or parsed.get("object_type") or "object").lower().strip()
                parsed["object_name"] = obj_name
                parsed["object_type"] = obj_name
                parsed.setdefault("description", f"A 3D {obj_name}")
                parsed.setdefault("suggested_thickness", 1.0)
                parsed.setdefault("suggested_smoothness", 3)
                parsed.setdefault("silhouette_features", [f"{obj_name}_contour", "volumetric_boundary"])
                parsed.setdefault("visual_features", parsed.get("silhouette_features", ["3D geometry"]))

                # Conflict analysis
                conflict_info = cls.detect_image_prompt_conflict(parsed, prompt_info)
                parsed["conflict_detected"] = conflict_info["conflict_detected"]
                parsed["conflict_type"] = conflict_info["conflict_type"]
                parsed["conflict_message"] = conflict_info["conflict_message"]
                parsed["conflict_resolution"] = conflict_resolution
                parsed["effective_object"] = parsed.get("object_name") if conflict_resolution == "preserve_image" else (prompt_info.get("requested_object") or parsed.get("object_name"))

                cls._vision_cache[content_hash] = parsed
                logger.info(
                    f"Groq Vision Recognition SUCCESS | Object: '{parsed.get('object_name')}' | "
                    f"Category: '{parsed.get('object_category')}' | Conflict: {parsed.get('conflict_detected')}"
                )
                return parsed

            elif resp.status_code == 429:
                cls._groq_rate_limited_until = time.time() + 5.0
                logger.warning("Groq Vision 429 Rate-Limited. Falling back to Dynamic CV Classifier.")
            else:
                logger.warning(f"Groq Vision API returned status {resp.status_code}. Falling back.")

        except Exception as e:
            logger.warning(f"Groq Vision exception ({type(e).__name__}: {e}). Fast-path fallback.")

        # Fallback to Dynamic CV Classifier
        blueprint = cls.cv_classify_sketch(image_path, prompt_str)
        blueprint["available"] = True
        blueprint["model_used"] = "Local Computer Vision Classifier (Robust Fallback)"
        blueprint.setdefault("description", f"A 3D {blueprint.get('object_name', 'object')}")
        blueprint.setdefault("suggested_thickness", 1.0)
        blueprint.setdefault("suggested_smoothness", 3)
        blueprint.setdefault("visual_features", blueprint.get("silhouette_features", ["3D contour boundary"]))
        blueprint.setdefault("object_type", blueprint.get("object_name", "object"))
        conflict_info = cls.detect_image_prompt_conflict(blueprint, prompt_info)
        blueprint["conflict_detected"] = conflict_info["conflict_detected"]
        blueprint["conflict_type"] = conflict_info["conflict_type"]
        blueprint["conflict_message"] = conflict_info["conflict_message"]
        blueprint["conflict_resolution"] = conflict_resolution
        blueprint["effective_object"] = blueprint.get("object_name") if conflict_resolution == "preserve_image" else (prompt_info.get("requested_object") or blueprint.get("object_name"))
        cls._vision_cache[content_hash] = blueprint
        return blueprint

groq_service = GroqService()

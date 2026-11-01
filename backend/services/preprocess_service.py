import cv2
import numpy as np
import logging
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from backend.core.config import UPLOADS_DIR, PROCESSED_DIR

logger = logging.getLogger("preprocess_service")

class PreprocessService:
    @staticmethod
    def extract_solid_silhouette(binary_mask: np.ndarray) -> np.ndarray:
        """
        Extracts a solid silhouette from sketches (handling both filled shapes and line drawings).
        If the sketch is an outline drawing, enclosed regions are filled to represent the 3D volume,
        while genuine topological holes (e.g. mug handles, gaps between legs) are preserved.
        """
        _, bin_img = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

        # Detect inverted mask (if border is predominantly foreground, invert)
        border_pixels = np.concatenate([
            bin_img[0, :], bin_img[-1, :], bin_img[:, 0], bin_img[:, -1]
        ])
        if np.mean(border_pixels) > 180:
            bin_img = cv2.bitwise_not(bin_img)

        # Morphological closing to bridge small outline gaps in sketches
        ksize = max(5, int(bin_img.shape[0] * 0.02) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        # Extract solid external envelope of the drawn object
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours is None or len(contours) == 0:
            return bin_img

        canvas = np.zeros_like(bin_img)
        cv2.drawContours(canvas, contours, -1, 255, -1)

        # Always preserve original strokes so thin features are never lost
        solid = canvas | bin_img
        return solid

    @staticmethod
    def decompose_sketch_structure(solid_mask: np.ndarray, raw_strokes: np.ndarray) -> Dict[str, Any]:
        """
        Decomposes the 2D sketch into structured geometric sub-components:
        - Major topological holes (e.g. mug handle loop)
        - Main body mask vs. appendage/handle mask
        - Horizontal radius profile R(y) and center line x_c(y)
        - Aspect ratio (width / height)
        """
        h, w = solid_mask.shape
        contours, hierarchy = cv2.findContours(solid_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        holes = []
        if hierarchy is not None:
            for i, c in enumerate(contours):
                if hierarchy[0][i][3] != -1:
                    area = cv2.contourArea(c)
                    if area > 20:
                        hx, hy, hw, hh = cv2.boundingRect(c)
                        holes.append({
                            "area": float(area),
                            "bbox": (int(hx), int(hy), int(hw), int(hh)),
                            "center": (float(hx + hw / 2), float(hy + hh / 2)),
                            "contour": c
                        })

        body_mask = solid_mask.copy()
        handle_mask = np.zeros_like(solid_mask)
        has_side_handle = False
        handle_side = "none"

        ys, xs = np.where(solid_mask > 0)
        shape_cx = float((xs.min() + xs.max()) / 2.0) if len(xs) > 0 else float(w / 2.0)
        total_width = float(xs.max() - xs.min()) if len(xs) > 0 else 1.0
        total_height = float(ys.max() - ys.min()) if len(ys) > 0 else 1.0
        aspect_ratio = total_width / (total_height + 1e-6)

        # Only search for side handles on near-vertical rotational vessels (aspect ~ 0.55 to 1.45)
        # Vehicles, horizontal profiles, animals, furniture, phones NEVER have mug side handles
        is_potential_vessel = (0.55 <= aspect_ratio <= 1.45)

        if is_potential_vessel and len(holes) == 1 and len(xs) > 0:
            major_hole = holes[0]
            hx, hy, hw, hh = major_hole["bbox"]
            hole_cx = hx + hw / 2.0

            # Determine if hole is on left or right side of main body
            if hole_cx < shape_cx - total_width * 0.15:
                # Handle on left
                handle_side = "left"
                has_side_handle = True
                x_split = hx + int(hw * 0.8)
                y_top_h = max(0, hy - int(hh * 0.25))
                y_bot_h = min(h, hy + hh + int(hh * 0.25))
                handle_mask[y_top_h:y_bot_h, :x_split] = solid_mask[y_top_h:y_bot_h, :x_split]
                body_mask = (solid_mask > 0) & (~(handle_mask > 0))
            elif hole_cx > shape_cx + total_width * 0.15:
                # Handle on right
                handle_side = "right"
                has_side_handle = True
                x_split = hx + int(hw * 0.2)
                y_top_h = max(0, hy - int(hh * 0.25))
                y_bot_h = min(h, hy + hh + int(hh * 0.25))
                handle_mask[y_top_h:y_bot_h, x_split:] = solid_mask[y_top_h:y_bot_h, x_split:]
                body_mask = (solid_mask > 0) & (~(handle_mask > 0))

        # Calculate bounding box of main body
        body_ys, body_xs = np.where(body_mask > 0)
        if len(body_xs) == 0:
            body_mask = solid_mask.copy()
            handle_mask = np.zeros_like(solid_mask)
            body_ys, body_xs = np.where(body_mask > 0)

        y_top = int(body_ys.min()) if len(body_ys) > 0 else 0
        y_bot = int(body_ys.max()) if len(body_ys) > 0 else h - 1

        # Calculate vertical radius profile R(y) and center line x_c(y)
        r_profile = np.zeros(h, dtype=np.float32)
        x_center = np.zeros(h, dtype=np.float32)

        for y in range(y_top, y_bot + 1):
            row_xs = np.where(body_mask[y, :] > 0)[0]
            if len(row_xs) > 0:
                r_profile[y] = (row_xs.max() - row_xs.min()) / 2.0
                x_center[y] = (row_xs.max() + row_xs.min()) / 2.0

        total_width = float(xs.max() - xs.min()) if len(xs) > 0 else 1.0
        total_height = float(ys.max() - ys.min()) if len(ys) > 0 else 1.0
        aspect_ratio = total_width / (total_height + 1e-6)

        # Orientation & Principal Axis Analysis (PCA + Minimum Area Rect)
        is_elongated = False
        major_len = max(total_width, total_height)
        minor_width = min(total_width, total_height)
        tip_direction = "none"
        principal_angle = 0.0

        if len(xs) > 20:
            pts_f = np.column_stack([xs, ys]).astype(np.float32)
            try:
                rect = cv2.minAreaRect(pts_f)
                r_w, r_h = rect[1]
                rect_major = max(r_w, r_h)
                rect_minor = max(1.0, min(r_w, r_h))
                axial_ratio = rect_major / rect_minor
                if axial_ratio >= 2.2:
                    is_elongated = True
                    major_len = rect_major
                    minor_width = rect_minor

                # Compute PCA to find primary direction vector
                mean = np.mean(pts_f, axis=0)
                cov = np.cov(pts_f - mean, rowvar=False)
                vals, vecs = np.linalg.eigh(cov)
                p_dir = vecs[:, 1]
                principal_angle = float(np.degrees(np.arctan2(p_dir[1], p_dir[0])))

                # Determine if one end has a tapered tip (pencil tip, pen point, needle)
                # Orient vector along primary axis
                if abs(p_dir[1]) >= abs(p_dir[0]):
                    # Primarily vertical: make p_dir point downwards (+Y)
                    if p_dir[1] < 0:
                        p_dir = -p_dir
                    proj = (pts_f - mean) @ p_dir
                    p_min, p_max = proj.min(), proj.max()
                    p_span = max(1.0, p_max - p_min)
                    a_start = np.sum(proj < p_min + 0.12 * p_span)
                    a_end = np.sum(proj > p_max - 0.12 * p_span)
                    if a_start < a_end * 0.65:
                        tip_direction = "top"
                    elif a_end < a_start * 0.65:
                        tip_direction = "bottom"
                else:
                    # Primarily horizontal: make p_dir point rightwards (+X)
                    if p_dir[0] < 0:
                        p_dir = -p_dir
                    proj = (pts_f - mean) @ p_dir
                    p_min, p_max = proj.min(), proj.max()
                    p_span = max(1.0, p_max - p_min)
                    a_start = np.sum(proj < p_min + 0.12 * p_span)
                    a_end = np.sum(proj > p_max - 0.12 * p_span)
                    if a_start < a_end * 0.65:
                        tip_direction = "left"
                    elif a_end < a_start * 0.65:
                        tip_direction = "right"
            except Exception as e:
                logger.warning(f"Orientation analysis fallback: {e}")

        # Horizontal vs. Vertical Component Analysis (for furniture: seat, table top, legs)
        furniture_features = {
            "has_horizontal_slab": False,
            "slab_y_range": (0, 0),
            "leg_y_range": (0, 0),
            "leg_x_positions": []
        }
        if len(ys) > 50:
            try:
                row_widths = np.sum(solid_mask > 0, axis=1)
                active_rows = np.where(row_widths > 0)[0]
                if len(active_rows) > 10:
                    max_w_row = int(np.argmax(row_widths))
                    max_w = float(row_widths[max_w_row])
                    # Detect sudden width jump or continuous wide slab
                    slab_rows = np.where(row_widths >= max_w * 0.75)[0]
                    if len(slab_rows) > 3:
                        s_min, s_max = int(slab_rows.min()), int(slab_rows.max())
                        furniture_features["has_horizontal_slab"] = True
                        furniture_features["slab_y_range"] = (s_min, s_max)
                        # Leg range below slab
                        legs_mask = solid_mask[s_max:, :]
                        if np.sum(legs_mask > 0) > 20:
                            furniture_features["leg_y_range"] = (s_max, int(ys.max()))
                            col_sums = np.sum(legs_mask > 0, axis=0)
                            leg_cols = np.where(col_sums > 0)[0]
                            if len(leg_cols) > 0:
                                furniture_features["leg_x_positions"] = [int(leg_cols.min()), int(leg_cols.max())]
            except Exception as e:
                logger.warning(f"Furniture structural analysis fallback: {e}")

        return {
            "has_side_handle": has_side_handle,
            "handle_side": handle_side,
            "body_mask": ((body_mask > 0).astype(np.uint8) * 255),
            "handle_mask": ((handle_mask > 0).astype(np.uint8) * 255),
            "holes": holes,
            "y_top": y_top,
            "y_bottom": y_bot,
            "radius_profile": r_profile,
            "center_profile": x_center,
            "aspect_ratio": round(aspect_ratio, 3),
            "is_elongated": is_elongated,
            "major_length": float(major_len),
            "minor_width": float(minor_width),
            "tip_direction": tip_direction,
            "principal_angle": round(principal_angle, 1),
            "furniture_features": furniture_features
        }

    @staticmethod
    def process_sketch(
        image_path: Path,
        threshold_val: int = 220,
        remove_bg: bool = True,
        invert_input: bool = False,
        do_edge_detect: bool = True,
        output_size: Tuple[int, int] = (512, 512)
    ) -> Dict[str, Any]:
        """
        Validates, filters, extracts contours, and prepares foreground mask and structural decomposition.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")

        pil_img = Image.open(image_path)
        pil_rgba = pil_img.convert("RGBA")
        np_rgba = np.array(pil_rgba)

        rgb = np_rgba[:, :, :3]
        alpha = np_rgba[:, :, 3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        if invert_input:
            gray = 255 - gray

        has_alpha_transparency = np.any(alpha < 240) and np.any(alpha > 15)

        if has_alpha_transparency and remove_bg:
            _, binary_mask = cv2.threshold(alpha, 30, 255, cv2.THRESH_BINARY)
        else:
            mean_intensity = np.mean(gray)
            if mean_intensity > 127:
                _, binary_mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
            else:
                _, binary_mask = cv2.threshold(gray, 255 - threshold_val, 255, cv2.THRESH_BINARY)

        # Solid silhouette
        solid_mask = PreprocessService.extract_solid_silhouette(binary_mask)

        # Uniform aspect-ratio preserving scaling into output_size with centered padding
        ys_occ, xs_occ = np.where(solid_mask > 0)
        if len(xs_occ) > 20:
            x_min, x_max = int(xs_occ.min()), int(xs_occ.max())
            y_min, y_max = int(ys_occ.min()), int(ys_occ.max())
            bbox_w = max(1, x_max - x_min + 1)
            bbox_h = max(1, y_max - y_min + 1)

            margin = 24
            avail_w = output_size[0] - 2 * margin
            avail_h = output_size[1] - 2 * margin
            scale = min(avail_w / bbox_w, avail_h / bbox_h)

            target_w = max(4, int(bbox_w * scale))
            target_h = max(4, int(bbox_h * scale))

            crop_solid = solid_mask[y_min:y_max+1, x_min:x_max+1]
            crop_raw = binary_mask[y_min:y_max+1, x_min:x_max+1]

            scaled_solid = cv2.resize(crop_solid, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            scaled_raw = cv2.resize(crop_raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

            resized_solid = np.zeros(output_size, dtype=np.uint8)
            resized_raw = np.zeros(output_size, dtype=np.uint8)

            off_x = (output_size[0] - target_w) // 2
            off_y = (output_size[1] - target_h) // 2

            resized_solid[off_y:off_y+target_h, off_x:off_x+target_w] = scaled_solid
            resized_raw[off_y:off_y+target_h, off_x:off_x+target_w] = scaled_raw

            preview_rgba = np.zeros((output_size[1], output_size[0], 4), dtype=np.uint8)
            crop_rgb = rgb[y_min:y_max+1, x_min:x_max+1]
            scaled_rgb = cv2.resize(crop_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            preview_rgba[off_y:off_y+target_h, off_x:off_x+target_w, :3] = scaled_rgb
            preview_rgba[off_y:off_y+target_h, off_x:off_x+target_w, 3] = scaled_solid
        else:
            resized_solid = cv2.resize(solid_mask, output_size, interpolation=cv2.INTER_NEAREST)
            resized_raw = cv2.resize(binary_mask, output_size, interpolation=cv2.INTER_NEAREST)
            preview_rgba = np.zeros((output_size[1], output_size[0], 4), dtype=np.uint8)
            resized_rgb = cv2.resize(rgb, output_size, interpolation=cv2.INTER_AREA)
            preview_rgba[:, :, :3] = resized_rgb
            preview_rgba[:, :, 3] = resized_solid

        # Structural decomposition
        structure = PreprocessService.decompose_sketch_structure(resized_solid, resized_raw)

        # Contours for metrics
        contours, _ = cv2.findContours(resized_solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)

        processed_id = f"proc_{image_path.stem}"
        processed_file = PROCESSED_DIR / f"{processed_id}.png"
        preview_file = PROCESSED_DIR / f"{processed_id}_preview.png"

        cv2.imwrite(str(processed_file), resized_solid)
        Image.fromarray(preview_rgba).save(preview_file)

        total_pixels = output_size[0] * output_size[1]
        active_pixels = np.sum(resized_solid > 0)
        coverage_ratio = float(active_pixels) / float(total_pixels)

        return {
            "processed_id": processed_id,
            "processed_path": processed_file,
            "preview_path": preview_file,
            "width": output_size[0],
            "height": output_size[1],
            "contours_detected": contour_count,
            "coverage_ratio": coverage_ratio,
            "structure": structure
        }

preprocess_service = PreprocessService()

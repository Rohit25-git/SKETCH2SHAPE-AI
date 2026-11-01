import trimesh
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from trimesh.visual.material import PBRMaterial

class MeshService:
    @staticmethod
    def calculate_metrics(geometry: Union[trimesh.Trimesh, trimesh.Scene]) -> Dict[str, Any]:
        """
        Calculates honest geometric metrics directly from the generated 3D mesh or Scene.
        """
        if isinstance(geometry, trimesh.Scene):
            geometries = list(geometry.geometry.values())
            extents = [float(x) for x in geometry.bounding_box.extents]
            v_count = sum(int(len(g.vertices)) for g in geometries)
            f_count = sum(int(len(g.faces)) for g in geometries)
            watertight = bool(all(getattr(g, "is_watertight", False) for g in geometries)) if geometries else False
            vol = sum(float(abs(getattr(g, "volume", 0.0))) for g in geometries if getattr(g, "is_watertight", False))
            area = sum(float(getattr(g, "area", 0.0)) for g in geometries)

            return {
                "vertex_count": v_count,
                "face_count": f_count,
                "is_watertight": watertight,
                "dimensions": extents,
                "volume": round(vol, 4),
                "surface_area": round(area, 4),
                "connected_components": len(geometries)
            }

        mesh = geometry
        extents = [float(x) for x in mesh.bounding_box.extents]
        try:
            vol = float(abs(mesh.volume)) if mesh.is_watertight else 0.0
        except Exception:
            vol = 0.0

        try:
            area = float(mesh.area)
        except Exception:
            area = 0.0

        components_count = 1
        try:
            if hasattr(mesh, "body_count"):
                components_count = int(mesh.body_count)
            elif len(mesh.faces) < 4000:
                components_count = len(mesh.split(only_watertight=False))
        except Exception:
            components_count = 1

        return {
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "is_watertight": bool(mesh.is_watertight),
            "dimensions": extents,
            "volume": round(vol, 4),
            "surface_area": round(area, 4),
            "connected_components": components_count
        }

    @staticmethod
    def apply_pbr_material(
        mesh: trimesh.Trimesh,
        material_props: Optional[Dict[str, Any]] = None
    ) -> trimesh.Trimesh:
        """
        Applies a physically based rendering (PBR) material to the mesh based on prompt/AI analysis.
        """
        if not material_props:
            material_props = {
                "base_color": [0.92, 0.94, 0.98, 1.0],
                "metallic": 0.05,
                "roughness": 0.20,
                "name": "GlossyCeramic"
            }

        color = material_props.get("base_color", [0.92, 0.94, 0.98, 1.0])
        # Ensure 4 float elements in [0, 1]
        if isinstance(color, list) and len(color) == 4 and any(c > 1.0 for c in color):
            color = [c / 255.0 for c in color]

        pbr = PBRMaterial(
            baseColorFactor=color,
            metallicFactor=float(material_props.get("metallic", 0.05)),
            roughnessFactor=float(material_props.get("roughness", 0.20)),
            name=str(material_props.get("name", "StudioMaterial"))
        )

        mesh.visual.material = pbr
        return mesh

    @staticmethod
    def export_glb_and_obj(
        geometry: Union[trimesh.Trimesh, trimesh.Scene],
        output_glb_path: Path,
        output_obj_path: Path,
        material_props: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Exports the mesh or scene into binary GLB with PBR materials and Wavefront OBJ files.
        """
        output_glb_path.parent.mkdir(parents=True, exist_ok=True)
        output_obj_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(geometry, trimesh.Scene):
            # Binary GLB export directly preserves named submeshes & distinct PBR materials
            glb_data = geometry.export(file_type="glb")
            with open(output_glb_path, "wb") as f:
                f.write(glb_data)

            # Wavefront OBJ export
            try:
                obj_data = geometry.export(file_type="obj")
                if isinstance(obj_data, str):
                    with open(output_obj_path, "w", encoding="utf-8") as f:
                        f.write(obj_data)
                else:
                    with open(output_obj_path, "wb") as f:
                        f.write(obj_data)
            except Exception:
                concat = trimesh.util.concatenate([g for g in geometry.geometry.values()])
                concat.export(str(output_obj_path), file_type="obj")

            verified_scene = trimesh.load(str(output_glb_path))
            if not verified_scene:
                raise RuntimeError("Exported GLB failed integrity verification")

            metrics = MeshService.calculate_metrics(geometry)
            metrics["glb_size_bytes"] = output_glb_path.stat().st_size
            metrics["obj_size_bytes"] = output_obj_path.stat().st_size
            return metrics

        # Single Trimesh mesh
        mesh = geometry
        has_vertex_colors = hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0
        if material_props and not has_vertex_colors and not hasattr(mesh.visual, "material"):
            MeshService.apply_pbr_material(mesh, material_props)

        # Export binary GLB (preserves COLOR_0 multi-color semantic vertex colors)
        mesh.export(str(output_glb_path), file_type="glb")

        # Export Wavefront OBJ
        mesh.export(str(output_obj_path), file_type="obj")

        # Verify export by reloading GLB
        verified_scene = trimesh.load(str(output_glb_path))
        if not verified_scene:
            raise RuntimeError("Exported GLB failed integrity verification")

        metrics = MeshService.calculate_metrics(mesh)
        metrics["glb_size_bytes"] = output_glb_path.stat().st_size
        metrics["obj_size_bytes"] = output_obj_path.stat().st_size
        return metrics

mesh_service = MeshService()

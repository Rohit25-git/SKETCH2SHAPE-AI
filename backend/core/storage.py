import os
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from backend.core.config import UPLOADS_DIR, PROCESSED_DIR, MODELS_DIR, METADATA_DIR

class StorageManager:
    @staticmethod
    def get_safe_filename(original_filename: str) -> str:
        ext = Path(original_filename).suffix.lower()
        unique_id = str(uuid.uuid4())
        return f"{unique_id}{ext}"

    @staticmethod
    def save_upload_bytes(data: bytes, original_filename: str) -> tuple[str, Path]:
        filename = StorageManager.get_safe_filename(original_filename)
        dest_path = UPLOADS_DIR / filename
        with open(dest_path, "wb") as f:
            f.write(data)
        return filename, dest_path

    @staticmethod
    def save_metadata(model_id: str, metadata: Dict[str, Any]) -> Path:
        meta_path = METADATA_DIR / f"{model_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return meta_path

    @staticmethod
    def load_metadata(model_id: str) -> Optional[Dict[str, Any]]:
        meta_path = METADATA_DIR / f"{model_id}.json"
        if not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def list_all_metadata() -> List[Dict[str, Any]]:
        items = []
        for file in METADATA_DIR.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    items.append(data)
            except Exception:
                continue
        # Sort descending by creation timestamp
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return items

    @staticmethod
    def delete_model_entry(model_id: str) -> bool:
        meta = StorageManager.load_metadata(model_id)
        if not meta:
            return False

        # Remove associated files
        files_to_remove = [
            MODELS_DIR / f"{model_id}.glb",
            MODELS_DIR / f"{model_id}.obj",
            MODELS_DIR / f"{model_id}.mtl",
            METADATA_DIR / f"{model_id}.json"
        ]
        
        # If upload or processed files exist and are unique to this model
        if "upload_path" in meta and meta["upload_path"]:
            p = Path(meta["upload_path"])
            if p.exists():
                files_to_remove.append(p)
        if "processed_path" in meta and meta["processed_path"]:
            p = Path(meta["processed_path"])
            if p.exists():
                files_to_remove.append(p)

        deleted = False
        for f in files_to_remove:
            try:
                if f.exists():
                    f.unlink()
                    deleted = True
            except Exception:
                pass
        return deleted

storage_manager = StorageManager()

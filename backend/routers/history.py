from typing import List
from fastapi import APIRouter, HTTPException
from backend.core.storage import storage_manager
from backend.models.schemas import HistoryItem

router = APIRouter()

@router.get("/history", response_model=List[HistoryItem])
async def get_generation_history():
    """
    Returns persistent records of previously generated 3D models.
    """
    records = storage_manager.list_all_metadata()
    history = []
    for item in records:
        geo = item.get("geometry", {})
        history.append(
            HistoryItem(
                model_id=item["model_id"],
                prompt=item.get("prompt"),
                engine=item.get("engine", "local_manifold"),
                created_at=item.get("created_at", ""),
                vertex_count=geo.get("vertex_count", 0),
                face_count=geo.get("face_count", 0),
                preview_image_url=item.get("preview_image_url", ""),
                glb_url=item.get("glb_url", f"/api/model/{item['model_id']}/download?format=glb"),
                obj_url=item.get("obj_url", f"/api/model/{item['model_id']}/download?format=obj")
            )
        )
    return history

@router.delete("/history/{model_id}")
async def delete_history_item(model_id: str):
    """
    Safely deletes model entry and associated 3D/image files from local storage.
    """
    success = storage_manager.delete_model_entry(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model ID '{model_id}' not found or already deleted")
    return {"status": "deleted", "model_id": model_id}

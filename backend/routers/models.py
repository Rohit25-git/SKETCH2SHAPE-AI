from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from backend.core.config import MODELS_DIR
from backend.core.storage import storage_manager
from backend.models.schemas import ModelMetadataResponse

router = APIRouter()

@router.get("/model/{model_id}", response_model=ModelMetadataResponse)
async def get_model_details(model_id: str):
    """
    Returns verified geometry metrics and download references for a generated 3D asset.
    """
    metadata = storage_manager.load_metadata(model_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    return ModelMetadataResponse(
        model_id=metadata["model_id"],
        job_id=metadata.get("job_id"),
        prompt=metadata.get("prompt"),
        engine=metadata.get("engine", "local_manifold"),
        created_at=metadata.get("created_at", ""),
        processing_time_seconds=metadata.get("processing_time_seconds", 0.0),
        geometry=metadata["geometry"],
        ai_analysis=metadata.get("ai_analysis"),
        preview_image_url=metadata.get("preview_image_url", ""),
        glb_url=metadata.get("glb_url", f"/api/model/{model_id}/download?format=glb"),
        obj_url=metadata.get("obj_url", f"/api/model/{model_id}/download?format=obj")
    )

@router.get("/model/{model_id}/download")
async def download_model_file(
    model_id: str,
    format: str = Query("glb", pattern="^(glb|obj)$")
):
    """
    Downloads the genuine 3D file in binary GLB or wavefront OBJ format.
    """
    filename = f"{model_id}.{format}"
    file_path = MODELS_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"3D file '{filename}' does not exist on disk")

    media_type = "model/gltf-binary" if format == "glb" else "text/plain"
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    return FileResponse(
        path=str(file_path),
        filename=f"sketch2shape_{model_id}.{format}",
        media_type=media_type,
        headers=headers
    )

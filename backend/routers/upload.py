import base64
import io
from pathlib import Path
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.core.config import settings
from backend.core.storage import storage_manager
from backend.models.schemas import UploadResponse, Base64UploadRequest

router = APIRouter()

@router.post("/sketch/upload", response_model=UploadResponse)
async def upload_sketch_file(file: UploadFile = File(...)):
    """
    Accepts image file upload (PNG, JPG, WEBP) via multipart/form-data.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '{ext}'. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    contents = await file.read()
    if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum upload size of {settings.MAX_UPLOAD_SIZE_BYTES / (1024*1024):.1f}MB"
        )

    try:
        pil_img = Image.open(io.BytesIO(contents))
        w, h = pil_img.size
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is corrupted or not a valid image")

    filename, path = storage_manager.save_upload_bytes(contents, file.filename)
    image_id = Path(filename).stem

    return UploadResponse(
        image_id=image_id,
        filename=filename,
        url=f"/api/uploads/{filename}",
        width=w,
        height=h,
        size_bytes=len(contents)
    )

@router.post("/sketch/upload/base64", response_model=UploadResponse)
def upload_sketch_base64(payload: Base64UploadRequest):
    """
    Accepts raw base64 canvas drawing data via application/json.
    Synchronous execution prevents any ASGI TestClient event-loop contention.
    """
    data = payload.image_base64
    if not data:
        raise HTTPException(status_code=400, detail="image_base64 field is required")

    if "base64," in data:
        data = data.split("base64,")[1]
    try:
        binary_data = base64.b64decode(data)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")

    if len(binary_data) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Base64 image exceeds maximum size")

    try:
        pil_img = Image.open(io.BytesIO(binary_data))
        w, h = pil_img.size
    except Exception:
        raise HTTPException(status_code=400, detail="Base64 payload does not represent a valid image")

    filename, path = storage_manager.save_upload_bytes(binary_data, payload.filename or "sketch.png")
    image_id = Path(filename).stem

    return UploadResponse(
        image_id=image_id,
        filename=filename,
        url=f"/api/uploads/{filename}",
        width=w,
        height=h,
        size_bytes=len(binary_data)
    )

from pathlib import Path
from fastapi import APIRouter, HTTPException
from backend.core.config import UPLOADS_DIR
from backend.models.schemas import PreprocessRequest, PreprocessResponse, AIAnalysisRequest, AIAnalysisResponse
from backend.services.preprocess_service import preprocess_service
from backend.services.groq_service import groq_service

router = APIRouter()

def find_upload_file(image_id: str) -> Path:
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        candidate = UPLOADS_DIR / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    candidate = UPLOADS_DIR / image_id
    if candidate.exists():
        return candidate
    raise HTTPException(status_code=404, detail=f"Image ID '{image_id}' not found")

@router.post("/image/preprocess", response_model=PreprocessResponse)
async def preprocess_image(request: PreprocessRequest):
    """
    Applies image enhancement, background separation, edge detection, and contour extraction.
    """
    upload_path = find_upload_file(request.image_id)

    try:
        result = preprocess_service.process_sketch(
            image_path=upload_path,
            threshold_val=request.threshold or 220,
            remove_bg=request.remove_background if request.remove_background is not None else True,
            invert_input=request.invert or False,
            do_edge_detect=request.edge_detection if request.edge_detection is not None else True
        )
        proc_filename = result["processed_path"].name
        prev_filename = result["preview_path"].name

        return PreprocessResponse(
            processed_id=result["processed_id"],
            url=f"/api/processed/{proc_filename}",
            preview_url=f"/api/processed/{prev_filename}",
            width=result["width"],
            height=result["height"],
            contours_detected=result["contours_detected"],
            coverage_ratio=result["coverage_ratio"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

@router.post("/sketch/analyze", response_model=AIAnalysisResponse)
async def analyze_sketch_with_ai(request: AIAnalysisRequest):
    """
    Analyzes the 2D sketch using Groq Multimodal Vision AI (qwen/qwen3.8-27b).
    Extracts semantic object classification, structural visual features, and suggested 3D parameters.
    """
    upload_path = find_upload_file(request.image_id)
    analysis = await groq_service.analyze_sketch(upload_path, prompt_guidance=request.prompt)
    return AIAnalysisResponse(**analysis)

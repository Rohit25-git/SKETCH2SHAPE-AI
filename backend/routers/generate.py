import asyncio
from fastapi import APIRouter, BackgroundTasks, HTTPException
from backend.models.schemas import GenerateRequest, JobStatusResponse
from backend.services.job_manager import job_manager

router = APIRouter()

@router.post("/generate/3d", response_model=JobStatusResponse)
async def generate_3d_model(
    request: GenerateRequest,
    background_tasks: BackgroundTasks
):
    """
    Submits a genuine 3D reconstruction job and returns a real job tracker.
    Non-blocking async execution.
    """
    job_id = job_manager.create_job(
        image_id=request.image_id,
        prompt=request.prompt,
        engine=request.engine or "local_manifold",
        resolution=request.resolution or "standard",
        smoothness=request.smoothness or 3,
        thickness=request.thickness or 1.0,
        conflict_resolution=request.conflict_resolution or "preserve_image"
    )

    # Dispatch to background worker
    background_tasks.add_task(job_manager.run_pipeline, job_id)

    status = job_manager.get_job_status(job_id)
    return JobStatusResponse(**status)

@router.get("/generate/status/{job_id}", response_model=JobStatusResponse)
async def get_generation_status(job_id: str):
    """
    Polls real-time job stage, elapsed time, and completion status.
    """
    status = job_manager.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found")
    return JobStatusResponse(**status)

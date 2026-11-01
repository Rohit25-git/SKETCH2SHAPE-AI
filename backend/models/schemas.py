from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class UploadResponse(BaseModel):
    image_id: str
    filename: str
    url: str
    width: int
    height: int
    size_bytes: int

class Base64UploadRequest(BaseModel):
    image_base64: str
    filename: Optional[str] = "sketch.png"

class PreprocessRequest(BaseModel):
    image_id: str
    threshold: Optional[int] = Field(220, ge=0, le=255)
    remove_background: Optional[bool] = True
    invert: Optional[bool] = False
    edge_detection: Optional[bool] = True

class PreprocessResponse(BaseModel):
    processed_id: str
    url: str
    preview_url: str
    width: int
    height: int
    contours_detected: int
    coverage_ratio: float

class AIAnalysisRequest(BaseModel):
    image_id: str
    prompt: Optional[str] = None

class AIAnalysisResponse(BaseModel):
    available: bool
    object_type: str
    description: str
    suggested_thickness: float
    suggested_smoothness: int
    visual_features: List[str]
    object_category: Optional[str] = None
    object_name: Optional[str] = None
    main_body: Optional[Dict[str, Any]] = None
    components: Optional[List[Any]] = None
    symmetry: Optional[str] = None
    material: Optional[str] = None
    primary_color: Optional[str] = None
    roughness: Optional[float] = None
    metalness: Optional[float] = None
    confidence: Optional[float] = None
    model_used: Optional[str] = None
    error: Optional[str] = None
    conflict_detected: Optional[bool] = False
    conflict_type: Optional[str] = None
    conflict_message: Optional[str] = None
    conflict_resolution: Optional[str] = "preserve_image"
    effective_object: Optional[str] = None

class GenerateRequest(BaseModel):
    image_id: str
    prompt: Optional[str] = None
    engine: Optional[str] = "local_manifold"  # "local_manifold", "cloud_tripo", "cloud_meshy"
    resolution: Optional[str] = "standard"  # "fast", "standard", "high"
    smoothness: Optional[int] = Field(3, ge=1, le=10)
    thickness: Optional[float] = Field(1.0, ge=0.2, le=3.0)
    conflict_resolution: Optional[str] = "preserve_image"  # "preserve_image" or "override_with_prompt"

class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    stage: str   # 8 distinct honest stages
    stage_index: int
    total_stages: int = 8
    progress_percentage: int
    elapsed_seconds: float
    error: Optional[str] = None
    model_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class GeometryMetrics(BaseModel):
    vertex_count: int
    face_count: int
    is_watertight: bool
    dimensions: List[float] = Field(..., description="[width_x, height_y, depth_z]")
    volume: float
    surface_area: float

class ModelMetadataResponse(BaseModel):
    model_id: str
    job_id: Optional[str] = None
    prompt: Optional[str] = None
    engine: str
    created_at: str
    processing_time_seconds: float
    geometry: GeometryMetrics
    ai_analysis: Optional[Dict[str, Any]] = None
    preview_image_url: str
    glb_url: str
    obj_url: str

class HistoryItem(BaseModel):
    model_id: str
    prompt: Optional[str] = None
    engine: str
    created_at: str
    vertex_count: int
    face_count: int
    ai_object_type: Optional[str] = None
    preview_image_url: str
    glb_url: str
    obj_url: str

class HardwareInfo(BaseModel):
    os: str
    cpu_count: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    cuda_available: bool
    torch_version: Optional[str] = None

class SystemHealthResponse(BaseModel):
    status: str
    version: str
    hardware: HardwareInfo
    available_engines: List[str]
    cloud_api_configured: Dict[str, bool]
    groq_vision_configured: bool
    groq_vision_model: str

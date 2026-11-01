import platform
import psutil
from fastapi import APIRouter
from backend.core.config import settings
from backend.models.schemas import SystemHealthResponse, HardwareInfo
from backend.services.cloud_ai_service import cloud_ai_service
from backend.services.groq_service import groq_service

router = APIRouter()

# Detect once at startup to avoid re-checking on every request
_HW_INFO = None

def get_hardware_info() -> HardwareInfo:
    global _HW_INFO
    if _HW_INFO is None:
        vm = psutil.virtual_memory()
        _HW_INFO = HardwareInfo(
            os=platform.platform(),
            cpu_count=psutil.cpu_count(logical=True) or 1,
            ram_total_gb=round(vm.total / (1024 ** 3), 2),
            ram_available_gb=round(vm.available / (1024 ** 3), 2),
            gpu_available=False,
            gpu_name=None,
            cuda_available=False,
            torch_version=None
        )
    else:
        # Just refresh available memory
        try:
            vm = psutil.virtual_memory()
            _HW_INFO.ram_available_gb = round(vm.available / (1024 ** 3), 2)
        except Exception:
            pass
    return _HW_INFO

@router.get("/health", response_model=SystemHealthResponse)
def health_check():
    """
    Returns system status, genuine hardware metrics, and available reconstruction backends.
    """
    hw_info = get_hardware_info()

    available_engines = ["local_manifold"]
    if cloud_ai_service.is_tripo_configured():
        available_engines.append("cloud_tripo")
    if cloud_ai_service.is_meshy_configured():
        available_engines.append("cloud_meshy")

    return SystemHealthResponse(
        status="healthy",
        version=settings.VERSION,
        hardware=hw_info,
        available_engines=available_engines,
        cloud_api_configured={
            "tripo": cloud_ai_service.is_tripo_configured(),
            "meshy": cloud_ai_service.is_meshy_configured()
        },
        groq_vision_configured=groq_service.is_configured(),
        groq_vision_model="qwen/qwen3.8-27b (Groq LPU Vision)" if groq_service.is_configured() else "None"
    )

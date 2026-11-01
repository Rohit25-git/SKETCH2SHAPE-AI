import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
PROCESSED_DIR = STORAGE_DIR / "processed"
MODELS_DIR = STORAGE_DIR / "models"
METADATA_DIR = STORAGE_DIR / "metadata"

# Load .env file if present
env_file = BASE_DIR / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k not in os.environ:
                    os.environ[k] = v

for directory in [STORAGE_DIR, UPLOADS_DIR, PROCESSED_DIR, MODELS_DIR, METADATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class Settings(BaseModel):
    PROJECT_NAME: str = "SKETCH2SHAPE AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS: set = {".png", ".jpg", ".jpeg", ".webp"}
    CORS_ORIGINS: list = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ]
    # Generative AI & Vision API Key (Groq LPU Multimodal Vision)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Optional Cloud 3D Mesh Provider credentials
    TRIPO_API_KEY: str = os.getenv("TRIPO_API_KEY", "")
    MESHY_API_KEY: str = os.getenv("MESHY_API_KEY", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    DEFAULT_ENGINE: str = os.getenv("DEFAULT_ENGINE", "local_manifold")

settings = Settings()

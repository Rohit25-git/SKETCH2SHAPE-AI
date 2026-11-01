from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.core.config import settings, UPLOADS_DIR, PROCESSED_DIR, MODELS_DIR
from backend.routers import upload, preprocess, generate, models, history, health

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="SKETCH2SHAPE AI: From a simple sketch to a dimensional reality."
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file mount points for images and models
app.mount("/api/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/api/processed", StaticFiles(directory=str(PROCESSED_DIR)), name="processed")
app.mount("/api/models/static", StaticFiles(directory=str(MODELS_DIR)), name="models")

# Include Routers
app.include_router(health.router, prefix="/api", tags=["System"])
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(preprocess.router, prefix="/api", tags=["Preprocess"])
app.include_router(generate.router, prefix="/api", tags=["Reconstruction"])
app.include_router(models.router, prefix="/api", tags=["Models"])
app.include_router(history.router, prefix="/api", tags=["History"])

@app.get("/")
def root():
    return {
        "app": settings.PROJECT_NAME,
        "tagline": "From a simple sketch to a dimensional reality.",
        "docs": "/docs",
        "health": "/api/health"
    }

# SKETCH2SHAPE AI - System Architecture
## TECHZEPHYR 4.0 Specification & Technical Design

```
                             +-----------------------------+
                             |   React + Vite Frontend     |
                             |  Tailwind CSS + Three.js    |
                             +--------------+--------------+
                                            |
                                  REST APIs / JSON / Multipart
                                            |
                             +--------------v--------------+
                             |   FastAPI Backend Server    |
                             |   (Async Background Tasks)  |
                             +-------+---------------+-----+
                                     |               |
                   +-----------------+               +-----------------+
                   |                                                   |
        +----------v----------+                             +----------v----------+
        | Computer Vision &   |                             | Persistent Vault    |
        | Groq LPU Vision AI  |                             | - storage/uploads   |
        | - OpenCV & Pillow   |                             | - storage/processed |
        | - Groq Qwen 3.8-27B |                             | - storage/models    |
        | - Canny & Contours  |                             | - storage/metadata  |
        | - Crease Extraction |                             |                     |
        +----------+----------+                             +----------+----------+
                   |                                                   |
        +----------v---------------------------------------------------v----------+
        |                 3D Reconstruction Engine Pipeline                       |
        |                                                                         |
        |  [AI VISION] Groq LPU Multimodal Vision (`qwen/qwen3.8-27b`):           |
        |     - Active when `GROQ_API_KEY` is configured in `.env`                |
        |     - High-speed visual feature extraction & semantic classification    |
        |     - Recommended geometric thickness and mesh smoothness tuning        |
        |                                                                         |
        |  [ENGINE 1] Procedural Manifold SDF Engine (Offline / Local):           |
        |     - Euclidean Distance Transform (EDT) for continuous thickness       |
        |     - Internal crease relief modulation from original sketch strokes    |
        |     - 3D Signed Distance Field (SDF) voxel volume construction          |
        |     - Marching Cubes isosurface extraction (`skimage.measure`)          |
        |     - Trimesh Laplacian smoothing & normal optimization                 |
        |     - Verified watertight manifold mesh generation                      |
        |                                                                         |
        |  [ENGINE 2] Generative Neural AI Engine (Tripo3D API):                  |
        |     - Active when `TRIPO_API_KEY` is configured in `.env`               |
        |     - Single-image to multi-view diffusion and Tripo LRM transformer   |
        |     - Ingests sketch and natural-language text guidance                 |
        +----------------------------------+--------------------------------------+
                                           |
                                +----------v----------+
                                | Mesh Validation &   |
                                | Metric Computation  |
                                | - Vertex/Face Count |
                                | - Watertightness    |
                                | - Volume & Bounds   |
                                +----------+----------+
                                           |
                                +----------v----------+
                                | Exporters:          |
                                | - Binary GLB (.glb) |
                                | - Wavefront (.obj)  |
                                +---------------------+
```

## REST API Specification

| Endpoint | Method | Input | Response | Purpose |
|---|---|---|---|---|
| `/api/health` | GET | None | `SystemHealthResponse` | Hardware metrics, CPU cores, RAM, GPU/CUDA detection, engine & Groq status |
| `/api/sketch/upload` | POST | Multipart file (`PNG`/`JPG`/`WEBP`) | `UploadResponse` | Ingests sketch file, validates headers and size |
| `/api/sketch/upload/base64` | POST | JSON `Base64UploadRequest` | `UploadResponse` | Ingests raw canvas base64 image data |
| `/api/sketch/analyze` | POST | JSON `AIAnalysisRequest` | `AIAnalysisResponse` | Live Groq LPU Multimodal Vision (`qwen/qwen3.8-27b`) semantic feature extraction |
| `/api/image/preprocess` | POST | JSON `PreprocessRequest` | `PreprocessResponse` | OpenCV filtering, contour count, coverage ratio, preview URL |
| `/api/generate/3d` | POST | JSON `GenerateRequest` | `JobStatusResponse` | Spawns non-blocking async reconstruction pipeline |
| `/api/generate/status/{job_id}` | GET | Path `job_id` | `JobStatusResponse` | Real-time 8-stage progress tracker and elapsed time |
| `/api/model/{model_id}` | GET | Path `model_id` | `ModelMetadataResponse` | Verified geometry metrics, dimensions, volume, download URLs |
| `/api/model/{model_id}/download` | GET | Query `format=glb\|obj` | `FileResponse` | Direct binary stream of verified GLB or OBJ asset |
| `/api/history` | GET | None | `List[HistoryItem]` | All previously generated models with preview thumbnails |
| `/api/history/{model_id}` | DELETE | Path `model_id` | `{"status": "deleted"}` | Safe deletion of model files and metadata from disk |

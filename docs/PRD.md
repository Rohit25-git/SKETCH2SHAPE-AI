# SKETCH2SHAPE AI - Product Requirements Document (PRD)
## TECHZEPHYR 4.0 Official Problem Statement Compliance

### 1. Executive Summary & Problem Statement
- **Competition Reference:** TECHZEPHYR 4.0 (IIT Bhubaneswar)
- **Product Name:** SKETCH2SHAPE AI
- **Tagline:** "From a simple sketch to a dimensional reality."
- **Official Problem Expectation:** An end-to-end computer vision and 3D reconstruction system that accepts 2D sketches or uploaded drawings, extracts structural features, supports optional natural-language guidance, performs 3D geometry reconstruction, allows real-time interactive inspection, and exports standard, production-ready 3D assets (GLB / OBJ).

### 2. Official 100-Point Evaluation Rubric
1. **Technical Implementation (25 Points):** Robust backend architecture (FastAPI), computer vision stroke/contour extraction (OpenCV/Pillow), Groq LPU Multimodal Vision AI (`qwen/qwen3.8-27b`), non-blocking asynchronous pipeline, and verifiable geometry generation.
2. **3D Output Quality and Design (25 Points):** Genuine 3D manifold geometry, plausible depth and volume, recognizable fidelity to input sketch, surface coherence, vertex/normal calculation, and absence of flat billboard planes or fake textures.
3. **End-to-End Pipeline Validity and Efficiency (20 Points):** Honest 8-stage progress tracking without fabricated timers, smooth input-to-output workflow, multi-angle Three.js studio inspection, and GLB/OBJ download.
4. **Code Reproducibility (15 Points):** Clear virtual environment management, explicit hardware requirements, automated test suites (pytest + vitest), deterministic builds, and documented startup procedures.
5. **Documentation and README Quality (15 Points):** Comprehensive technical specifications, hardware/runtime benchmarking, architecture diagrams, API references, and honest attribution of procedural vs. generative AI capabilities.

### 3. AI & Reconstruction Architecture & Model Attribution
- **AI Vision Engine: Groq LPU Multimodal Vision (`qwen/qwen3.8-27b`)**
  - *Type:* Ultra-fast multimodal Large Vision-Language Model executed on Groq LPU hardware.
  - *Role:* Ingests sketch image base64, performs semantic object recognition, extracts visual structural features (rims, handles, silhouettes), and suggests optimal geometric reconstruction parameters (thickness and smoothness).
  - *Transparency:* Active when `GROQ_API_KEY` is provided in `.env`. Gracefully falls back to computer-vision heuristics if rate-limited or offline.
- **Engine 1: Procedural Manifold SDF Engine (Watertight Marching Cubes)**
  - *Type:* Deterministic Computer Vision & Geometric Inflation (Non-neural procedural engine).
  - *Technique:* Euclidean Distance Transform (EDT), Signed Distance Field (SDF) volumetric inflation, internal crease relief modulation, and Marching Cubes isosurface extraction (`skimage.measure.marching_cubes` + `trimesh`).
  - *Purpose:* Guaranteed local execution on constrained hardware (CPU or non-CUDA systems) producing verified watertight manifold meshes with zero external API dependencies.
- **Engine 2: Generative Neural AI Engine (Tripo3D API)**
  - *Type:* Deep Generative Large Reconstruction Model (LRM).
  - *Technique:* Single-image to multi-view diffusion and feed-forward transformer geometry generation.
  - *Configuration:* Enabled when `TRIPO_API_KEY` is provided in `.env`.
  - *Transparency:* If the API key is not configured, the system transparently reports that cloud generative AI is unavailable and uses the Procedural Manifold SDF engine without claiming procedural output as neural generative inference.

### 4. Functional Requirements Breakdown

#### 4.1 Computer Vision & Input Processing
- Multi-format ingestion: Multipart upload (`PNG`, `JPG`, `WEBP`) and raw HTML5 canvas base64 data.
- OpenCV preprocessing: Contrast normalization, Otsu & adaptive thresholding, sketch inversion detection, morphological opening/closing, Canny edge detection, and external contour extraction.
- Feature extraction: Contour count, coverage ratio, bounding box, stroke density, internal crease detection, and Groq Multimodal Vision feature detection.

#### 4.2 Interactive 2D Drawing Canvas
- High-DPI canvas supporting mouse and touch drawing.
- Adjustable brush and eraser with stroke width slider ($1\text{px} - 40\text{px}$).
- Studio color palette (Electric Violet, Cyan, White, Graphite, Neon) + custom color picker.
- Canvas background toggles: Pitch Black (`#0B0D13`), Blueprint Grid, Clean White.
- Multi-level Undo/Redo stack, clear canvas, and PNG export.

#### 4.3 Honest 8-Stage Progress Tracker
Non-blocking async state machine tracking:
1. `input_validation`: Resolution, format, and dimension verification.
2. `image_preprocessing`: OpenCV stroke enhancement, edge detection, and Groq Vision AI analysis.
3. `background_preparation`: Alpha segmentation and silhouette isolation.
4. `reconstruction_initialization`: Volumetric coordinate space and distance transform.
5. `3d_geometry_generation`: Volumetric SDF inflation and Marching Cubes.
6. `mesh_processing`: Normal computation, centering, and Laplacian smoothing.
7. `output_validation`: Boundary manifoldness and watertight metric validation.
8. `model_ready`: Asset compilation into GLB and OBJ.

#### 4.4 Real-Time 3D Studio & Geometry HUD
- WebGL Three.js / React Three Fiber interactive viewport.
- OrbitControls (rotate, pan, zoom, auto-rotate, camera reset).
- Viewport toggles: Wireframe overlay, Ground grid, Studio lighting.
- Verified Geometry HUD: Detected AI classification, vertex count, triangle face count, bounding box dimensions $(X \times Y \times Z)$, render duration, watertight verification badge.
- Direct downloads for binary GLB (`.glb`) and wavefront OBJ (`.obj`).

#### 4.5 Persistent Asset Vault (My Creations)
- Instant gallery display of all previously generated models.
- Thumbnail previews, detected semantic label, vertex count, date generated.
- Quick actions: Re-open in 3D studio, direct GLB download, safe asset deletion.

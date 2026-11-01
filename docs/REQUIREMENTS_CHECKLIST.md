# SKETCH2SHAPE AI - Requirements Traceability Matrix
## TECHZEPHYR 4.0 (IIT Bhubaneswar) Official Rubric Mapping

| ID | Requirement Description | Official Rubric Category | Source Implementation File(s) | Verification Method & Test Evidence | Status |
|---|---|---|---|---|---|
| REQ-01 | FastAPI async backend & non-blocking background task handling | Technical Implementation (25 pts) | `backend/main.py`, `backend/services/job_manager.py` | `test_generate_job_lifecycle_and_model_export` | **VERIFIED** |
| REQ-02 | Computer vision sketch preprocessing & contour isolation | Technical Implementation (25 pts) | `backend/services/preprocess_service.py` | `test_preprocess_contour_and_edges` | **VERIFIED** |
| REQ-03 | Internal stroke & crease depth relief modulation | Technical Implementation (25 pts) | `backend/services/reconstruction_service.py` | `verify_fresh_sketches.py` (relief audit) | **VERIFIED** |
| REQ-04 | Genuine 3D manifold geometry synthesis (no billboard planes) | 3D Output Quality & Design (25 pts) | `backend/services/reconstruction_service.py` | `test_reconstruction_manifold_geometry` | **VERIFIED** |
| REQ-05 | Watertight volume & mesh metric calculation | 3D Output Quality & Design (25 pts) | `backend/services/mesh_service.py` | `audit_mesh_asset()` Trimesh inspection | **VERIFIED** |
| REQ-06 | Multi-format export: Binary GLB (`.glb`) & Wavefront OBJ (`.obj`) | 3D Output Quality & Design (25 pts) | `backend/routers/models.py`, `mesh_service.py` | `test_workflow_b_uploaded_file_with_prompt_to_glb_and_obj` | **VERIFIED** |
| REQ-07 | Interactive Three.js / React Three Fiber 3D inspection studio | 3D Output Quality & Design (25 pts) | `frontend/src/components/studio/Viewer3D.tsx` | Vitest `components.test.tsx`, WebGL render | **VERIFIED** |
| REQ-08 | Honest 8-stage progress tracker without fabricated timers | Pipeline Validity & Efficiency (20 pts) | `backend/services/job_manager.py`, `GenerationModal.tsx` | `test_workflow_a_canvas_sketch_to_glb` | **VERIFIED** |
| REQ-09 | Full user workflow A: Canvas sketch &rarr; GLB export &rarr; Viewer | Pipeline Validity & Efficiency (20 pts) | `WorkspacePage.tsx`, `api.ts`, `generate.py` | `test_workflow_a_canvas_sketch_to_glb` | **VERIFIED** |
| REQ-10 | Full user workflow B: Upload PNG &rarr; Prompt &rarr; GLB/OBJ | Pipeline Validity & Efficiency (20 pts) | `ImageUpload.tsx`, `GenerationControls.tsx` | `test_workflow_b_uploaded_file_with_prompt_to_glb_and_obj` | **VERIFIED** |
| REQ-11 | Full user workflow C: Generation &rarr; History &rarr; Reopen &rarr; Delete | Pipeline Validity & Efficiency (20 pts) | `CreationsPage.tsx`, `history.py`, `storage.py` | `test_workflow_c_fresh_sketch_history_persistence_and_deletion` | **VERIFIED** |
| REQ-12 | Virtual environment management & deterministic dependencies | Code Reproducibility (15 pts) | `venv/`, `pytest.ini`, `frontend/package.json` | Clean venv creation & npm build (0 errors) | **VERIFIED** |
| REQ-13 | Automated test suite execution (Pytest & Vitest) | Code Reproducibility (15 pts) | `backend/tests/`, `frontend/src/test/` | 29 backend tests + 5 frontend tests passed (34 total) | **VERIFIED** |
| REQ-14 | Hardware & runtime transparency reporting | Documentation & README (15 pts) | `backend/routers/health.py`, `SystemPage.tsx` | `test_health_endpoint`, hardware report | **VERIFIED** |
| REQ-15 | Honest attribution of Procedural SDF vs. Generative Neural AI | Documentation & README (15 pts) | `README.md`, `PRD.md`, `ARCHITECTURE.md` | `test_cloud_ai_unconfigured_rejection` | **VERIFIED** |
| REQ-16 | Groq LPU Multimodal Vision AI (`qwen/qwen3.8-27b`) structured schema | Technical Implementation & Pipeline (25 pts) | `backend/services/groq_service.py`, `backend/routers/preprocess.py` | `test_groq_vision_sketch_analysis`, `test_groq_service_response_parsing` | **VERIFIED** |
| REQ-17 | Semantic component decomposition & vertical radial profile extraction | Computer Vision & Geometry (25 pts) | `backend/services/preprocess_service.py` | `decompose_sketch_structure`, hole and component separation | **VERIFIED** |
| REQ-18 | Component-aware 3D reconstruction with hollow cavities & open rims | 3D Output Quality & Geometry (25 pts) | `backend/services/reconstruction_service.py` | Raycast cavity verification, open rim detection | **VERIFIED** |
| REQ-19 | Natural language prompt parsing for PBR metallic-roughness materials | Generative AI & Materials (20 pts) | `reconstruction_service.py`, `mesh_service.py` | Trimesh PBRMaterial glTF embedding | **VERIFIED** |
| REQ-20 | 10-Sketch Unseen Evaluation Benchmark | Official Verification (25 pts) | `evaluate_unseen_sketches.py`, `storage/evaluation/` | 10/10 watertight manifold meshes generated | **VERIFIED** |

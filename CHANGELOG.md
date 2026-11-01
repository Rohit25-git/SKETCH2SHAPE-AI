# Changelog

All notable changes to the **SKETCH2SHAPE AI** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-11-01

### Added
- **Image-to-3D Local Manifold Reconstruction Pipeline**:
  - Distance Transform Field (DTF) & Signed Distance Field (SDF) voxel grid generator with Marching Cubes isosurface extraction (`scikit-image`, `scipy`).
  - Watertight 2-manifold volumetric mesh generation with Taubin smoothing, laplacian filtering, and normal recalculation (`trimesh`).
  - Architectural Building Procedural Synthesizer: 2-storey modular envelopes, gabled and dormer roof geometry, multi-pane fenestration windows, covered porch with columns, steps, and balustrade railings, and garage assemblies.
  - Vehicle Procedural Synthesizer: Sports car aerodynamics and Front-View Vintage Car synthesis with curved bonnet, 14-bar chrome radiator grille, circular headlights with glass lenses, front bumper with overriders, split windshield, and 4 white-wall rubber wheels with chrome hubcaps.
  - Organic Creature Synthesizer: Volumetric quadruped bodies, tapered craniums, pink snouts, ears, dual pinnae, articulated forelegs/hindlegs, paws, and curved tails.
  - Drinkware, Furniture, Electronics, Instruments, and Flora procedural synthesis pipelines.
- **Multimodal Vision & Object Identity Engine**:
  - Multimodal Vision integration (`qwen/qwen3.8-27b` on Groq LPU) providing real-time sketch interpretation.
  - Local Computer Vision Contour & Topology Classifier: dynamic aspect ratio, rectangular fenestration hole analysis, headlight detection, wheel separation, and symmetry analysis.
  - **Visual Identity Validation Gate & Image-First Policy**: Original sketch image strictly anchors 3D object geometry. User natural-language prompts refine materials, colors, and styles. Contradictory prompts trigger conflict warnings without overriding image identity.
- **PBR Photorealistic Materials & Textures**:
  - Intelligent multi-colour semantic component mapping.
  - Standard GLTF/GLB PBR material export with metallic-roughness workflow, diffuse base colors, and vertex colors.
  - Dual format export: binary GLTF (`.glb`) with embedded scene materials and Wavefront OBJ (`.obj`).
- **Interactive Three.js / React Studio**:
  - Viewport 3D canvas built on Three.js, `@react-three/fiber`, and `@react-three/drei`.
  - OrbitControls, grid ground plane, shadow maps, and directional fill lighting.
  - Material rendering switcher: PBR Photorealistic, Normals Shading, Wireframe Overlay.
  - **Pipeline Tracing & Diagnostics Drawer**: Live verification of Visual Identity Verification Gate, image SHA-256 hash, vision model used, prompt conditioning, provider task ID, and GLB hash.
  - 2D Canvas Drawing Studio with pencil, eraser, stroke width, and instant base64 export.
  - Interactive Preprocessing Preview with threshold tuning and Canny edge visualization.
  - Model history sidebar with instant reloading and deletion.

### Fixed
- **Vintage Car Mistaken for House Bug**: Fixed contour hole classification bug where radiator grille bars were falsely interpreted as architectural fenestrations.
- **Greyscale Mesh Issue**: Fixed missing Three.js material definitions and vertex color configurations in exported GLB scenes.
- **Text Prompt Identity Overriding**: Eliminated prompt hijacking where prompt text could replace uploaded image identity.

---

## [0.9.0] - 2026-10-31
- Initial prototype submission for TECHZEPHYR 4.0 Hackathon (IIT Bhubaneswar).
- FastAPI backend architecture with async job queue and REST endpoints.
- Integration test suite and baseline Marching Cubes pipeline.

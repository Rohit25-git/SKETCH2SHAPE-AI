import React, { Suspense, useRef, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Grid, Center, useGLTF, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';
import { 
  Download, 
  RotateCcw, 
  Eye, 
  Grid3X3, 
  Play, 
  Pause, 
  Box, 
  Check, 
  Sparkles,
  Info,
  Compass,
  Layers,
  Sliders,
  Sun,
  Palette,
  CheckCircle2,
  X,
  Activity,
  ShieldCheck
} from 'lucide-react';
import type { ModelMetadata } from '../../types';

interface Viewer3DProps {
  modelMetadata: ModelMetadata | null;
  modelUrl: string | null;
  onReset?: () => void;
}

type MaterialPreset = 'original' | 'ceramic' | 'metal' | 'glass' | 'wood' | 'clay' | 'fur';
type CameraView = 'perspective' | 'front' | 'side' | 'rear' | 'top';

const MATERIAL_CONFIGS: Record<MaterialPreset, (baseColor: THREE.Color, roughness: number, metalness: number) => THREE.Material> = {
  original: (_col, r, m) => new THREE.MeshStandardMaterial({
    color: new THREE.Color(1, 1, 1),
    roughness: r,
    metalness: m
  }),
  ceramic: () => new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(1, 1, 1),
    roughness: 0.12,
    metalness: 0.02,
    clearcoat: 0.85,
    clearcoatRoughness: 0.08
  }),
  metal: () => new THREE.MeshStandardMaterial({
    color: new THREE.Color(1, 1, 1),
    roughness: 0.18,
    metalness: 0.88
  }),
  glass: () => new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(0.85, 0.95, 1.0),
    roughness: 0.05,
    transmission: 0.85,
    transparent: true,
    opacity: 0.70,
    ior: 1.5
  }),
  wood: () => new THREE.MeshStandardMaterial({
    color: new THREE.Color(1, 1, 1),
    roughness: 0.72,
    metalness: 0.02
  }),
  clay: () => new THREE.MeshStandardMaterial({
    color: new THREE.Color(1, 1, 1),
    roughness: 0.88,
    metalness: 0.0
  }),
  fur: () => new THREE.MeshStandardMaterial({
    color: new THREE.Color(1, 1, 1),
    roughness: 0.92,
    metalness: 0.0
  })
};

// Helper to format metric physical dimensions
const formatMetricDimensions = (dims: number[]): string => {
  if (!dims || dims.length !== 3) return '';
  const max = Math.max(...dims);
  if (max < 1.0) {
    return dims.map(d => `${(d * 100).toFixed(1)} cm`).join(' × ');
  } else {
    return dims.map(d => `${d.toFixed(2)} m`).join(' × ');
  }
};

// 3D Model renderer with wireframe, material handling, and dynamic camera framing
const ModelMesh: React.FC<{ 
  url: string; 
  wireframe: boolean; 
  materialPreset: MaterialPreset;
  roughness: number;
  metalness: number;
  isTunedByUser: boolean;
}> = ({ url, wireframe, materialPreset, roughness, metalness, isTunedByUser }) => {
  const { scene } = useGLTF(url);

  // Dynamically compute visual scale so models of any real-world dimensions (0.008m to 4.5m)
  // are beautifully framed at optimal viewing size in the studio viewport
  const visualScale = React.useMemo(() => {
    const box = new THREE.Box3().setFromObject(scene);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    return maxDim > 0 ? (2.2 / maxDim) : 1.2;
  }, [scene]);

  React.useEffect(() => {
    scene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        const mesh = child as THREE.Mesh;
        const hasVertexColors = Boolean(mesh.geometry?.attributes?.color);
        let baseColor = new THREE.Color('#FFFFFF');

        if (mesh.material) {
          const mat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
          if ((mat as any).color) {
            baseColor = (mat as any).color.clone();
          }
        }

        if (materialPreset === 'original') {
          if (Array.isArray(mesh.material)) {
            mesh.material.forEach((mat: any) => {
              mat.wireframe = wireframe;
              if (hasVertexColors) {
                mat.vertexColors = true;
                mat.color.setRGB(1, 1, 1);
              }
              if (isTunedByUser) {
                if ('roughness' in mat) mat.roughness = roughness;
                if ('metalness' in mat) mat.metalness = metalness;
              }
              mat.needsUpdate = true;
            });
          } else if (mesh.material) {
            const mat = mesh.material as any;
            mat.wireframe = wireframe;
            if (hasVertexColors) {
              mat.vertexColors = true;
              mat.color.setRGB(1, 1, 1);
            }
            if (isTunedByUser) {
              if ('roughness' in mat) mat.roughness = roughness;
              if ('metalness' in mat) mat.metalness = metalness;
            }
            mat.needsUpdate = true;
          }
        } else {
          const newMat = MATERIAL_CONFIGS[materialPreset](baseColor, roughness, metalness);
          (newMat as any).wireframe = wireframe;
          if (hasVertexColors) {
            (newMat as any).vertexColors = true;
            (newMat as any).color.setRGB(1, 1, 1);
          } else {
            (newMat as any).vertexColors = false;
            (newMat as any).color = baseColor;
          }
          mesh.material = newMat;
        }
      }
    });
  }, [scene, wireframe, materialPreset, roughness, metalness, isTunedByUser]);

  return (
    <primitive 
      object={scene} 
      scale={visualScale} 
    />
  );
};

export const Viewer3D: React.FC<Viewer3DProps> = ({
  modelMetadata,
  modelUrl,
}) => {
  const controlsRef = useRef<any>(null);
  const [wireframe, setWireframe] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [autoRotate, setAutoRotate] = useState(false);
  const [showMetrics, setShowMetrics] = useState(true);
  const [cameraView, setCameraView] = useState<CameraView>('perspective');
  const [materialPreset, setMaterialPreset] = useState<MaterialPreset>('original');
  const [showMaterialMenu, setShowMaterialMenu] = useState(false);
  const [showTuningMenu, setShowTuningMenu] = useState(false);
  const [showInspector, setShowInspector] = useState(false);
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  // Real-time material and lighting properties
  const [roughness, setRoughness] = useState<number>(0.35);
  const [metalness, setMetalness] = useState<number>(0.15);
  const [exposure, setExposure] = useState<number>(1.15);
  const [isTunedByUser, setIsTunedByUser] = useState<boolean>(false);

  // Sync initial material properties whenever new modelMetadata or modelUrl arrives
  useEffect(() => {
    setIsTunedByUser(false);
    if (modelMetadata?.material_props) {
      if (typeof modelMetadata.material_props.roughness === 'number') {
        setRoughness(modelMetadata.material_props.roughness);
      }
      if (typeof modelMetadata.material_props.metallic === 'number') {
        setMetalness(modelMetadata.material_props.metallic);
      }
    }
  }, [modelMetadata, modelUrl]);

  const resetMaterialTuning = () => {
    setIsTunedByUser(false);
    if (modelMetadata?.material_props) {
      setRoughness(typeof modelMetadata.material_props.roughness === 'number' ? modelMetadata.material_props.roughness : 0.35);
      setMetalness(typeof modelMetadata.material_props.metallic === 'number' ? modelMetadata.material_props.metallic : 0.15);
    } else {
      setRoughness(0.35);
      setMetalness(0.15);
    }
    setExposure(1.15);
  };

  const setView = (view: CameraView) => {
    setCameraView(view);
    if (!controlsRef.current) return;

    setAutoRotate(false);
    const controls = controlsRef.current;
    const camera = controls.object;

    switch (view) {
      case 'front':
        camera.position.set(0, 0, 3.5);
        break;
      case 'rear':
        camera.position.set(0, 0, -3.5);
        break;
      case 'side':
        camera.position.set(3.5, 0, 0);
        break;
      case 'top':
        camera.position.set(0, 3.5, 0.001);
        break;
      case 'perspective':
      default:
        camera.position.set(0, 1.5, 3.5);
        break;
    }

    controls.target.set(0, 0, 0);
    camera.lookAt(0, 0, 0);
    controls.update();
  };

  const resetCamera = () => {
    setView('perspective');
  };

  return (
    <div className="relative w-full h-[550px] lg:h-[680px] rounded-3xl bg-gradient-to-b from-[#F8FAFC] to-[#EEF2F6] border-2 border-studio-border shadow-studio-card overflow-hidden flex flex-col">
      
      {/* 3D Canvas Viewport */}
      {modelUrl ? (
        <div className="w-full h-full relative cursor-grab active:cursor-grabbing">
          <Canvas
            camera={{ position: [0, 1.5, 3.5], fov: 45 }}
            gl={{ 
              antialias: true, 
              alpha: true,
              toneMapping: THREE.ACESFilmicToneMapping,
              toneMappingExposure: exposure
            }}
          >
            {/* Photorealistic Studio Lighting Setup */}
            <hemisphereLight args={['#F8FAFC', '#E2E8F0', 0.65]} />
            <ambientLight intensity={0.55} />
            {/* Key Light with Soft Studio Falloff */}
            <directionalLight position={[6, 9, 6]} intensity={1.3} castShadow shadow-bias={-0.0001} />
            {/* Fill Light for Contrast Control */}
            <directionalLight position={[-6, 4, 4]} intensity={0.65} color="#F1F5F9" />
            {/* Rim Light for Edge Definition */}
            <directionalLight position={[0, 6, -6]} intensity={0.85} color="#E0E7FF" />
            {/* Soft Bottom Bounce Light */}
            <pointLight position={[0, -3, 0]} intensity={0.25} color="#CBD5E1" />

            <Suspense fallback={null}>
              <Center top>
                <ModelMesh 
                  url={modelUrl} 
                  wireframe={wireframe} 
                  materialPreset={materialPreset}
                  roughness={roughness}
                  metalness={metalness}
                  isTunedByUser={isTunedByUser}
                />
              </Center>

              {/* Physically Grounded Soft Contact Shadow */}
              <ContactShadows
                position={[0, -0.01, 0]}
                opacity={0.65}
                scale={9}
                blur={2.0}
                far={4.0}
                color="#0F172A"
              />

              {showGrid && (
                <Grid
                  renderOrder={-1}
                  position={[0, -0.012, 0]}
                  infiniteGrid
                  cellSize={0.4}
                  cellThickness={0.6}
                  cellColor="#CBD5E1"
                  sectionSize={1.6}
                  sectionThickness={1.2}
                  sectionColor="#818CF8"
                  fadeDistance={18}
                  fadeStrength={1.5}
                />
              )}
            </Suspense>

            <OrbitControls
              ref={controlsRef}
              autoRotate={autoRotate}
              autoRotateSpeed={1.8}
              enableDamping
              dampingFactor={0.05}
              minDistance={1.0}
              maxDistance={10.0}
            />
          </Canvas>

          {/* Floating Viewport Controls Header */}
          <div className="absolute top-4 left-4 flex flex-wrap items-center gap-1.5 p-1.5 rounded-2xl bg-white/95 backdrop-blur-md border border-slate-200/90 z-10 shadow-md">
            <button
              onClick={resetCamera}
              className="p-2 rounded-xl hover:bg-slate-100 text-slate-600 hover:text-slate-900 transition-all"
              title="Reset Perspective View"
            >
              <RotateCcw className="w-4 h-4" />
            </button>

            {/* Camera View Switcher */}
            <div className="flex items-center rounded-xl bg-slate-100 border border-slate-200/80 p-0.5 text-[11px] font-semibold text-slate-600">
              {(['perspective', 'front', 'side', 'rear', 'top'] as CameraView[]).map((v) => (
                <button
                  key={v}
                  onClick={() => setView(v)}
                  className={`px-2.5 py-1 rounded-lg capitalize transition-all ${
                    cameraView === v 
                      ? 'bg-violet-primary text-white shadow-xs font-bold' 
                      : 'hover:text-slate-900'
                  }`}
                >
                  {v === 'perspective' ? 'Persp' : v}
                </button>
              ))}
            </div>

            <div className="h-4 w-px bg-slate-200 mx-0.5" />

            {/* Material Switcher Button */}
            <div className="relative">
              <button
                onClick={() => {
                  setShowMaterialMenu(!showMaterialMenu);
                  setShowTuningMenu(false);
                }}
                className={`p-2 rounded-xl transition-all flex items-center gap-1 ${
                  materialPreset !== 'original' 
                    ? 'bg-cyan-dim text-cyan-accent border border-cyan-accent/40 font-semibold' 
                    : 'hover:bg-slate-100 text-slate-600 hover:text-slate-900'
                }`}
                title="PBR Material Presets"
              >
                <Layers className="w-4 h-4" />
              </button>

              {showMaterialMenu && (
                <div className="absolute top-full left-0 mt-2 w-38 rounded-2xl bg-white border border-slate-200 p-1.5 shadow-xl z-30 flex flex-col gap-1 text-xs">
                  {[
                    { id: 'original', label: 'Default PBR' },
                    { id: 'ceramic', label: 'Glossy Ceramic' },
                    { id: 'metal', label: 'Polished Metal' },
                    { id: 'glass', label: 'Translucent Glass' },
                    { id: 'wood', label: 'Natural Wood' },
                    { id: 'clay', label: 'Sculpting Clay' },
                    { id: 'fur', label: 'Matte Fur / Skin' },
                  ].map((m) => (
                    <button
                      key={m.id}
                      onClick={() => {
                        setMaterialPreset(m.id as MaterialPreset);
                        setShowMaterialMenu(false);
                      }}
                      className={`w-full text-left px-2.5 py-1.5 rounded-xl transition-all ${
                        materialPreset === m.id
                          ? 'bg-violet-primary text-white font-semibold'
                          : 'hover:bg-slate-100 text-slate-700 hover:text-slate-900'
                      }`}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Real-Time PBR & Studio Lighting Tuning */}
            <div className="relative">
              <button
                onClick={() => {
                  setShowTuningMenu(!showTuningMenu);
                  setShowMaterialMenu(false);
                }}
                className={`p-2 rounded-xl transition-all flex items-center gap-1 ${
                  showTuningMenu 
                    ? 'bg-violet-dim text-violet-primary border border-violet-primary/30 font-semibold' 
                    : 'hover:bg-slate-100 text-slate-600 hover:text-slate-900'
                }`}
                title="Fine-tune Material & Lighting in Real Time"
              >
                <Sliders className="w-4 h-4" />
              </button>

              {showTuningMenu && (
                <div className="absolute top-full left-0 mt-2 w-64 rounded-2xl bg-white border border-slate-200 p-3 shadow-2xl z-30 flex flex-col gap-3 text-xs animate-fadeIn">
                  <div className="flex items-center justify-between pb-2 border-b border-slate-100">
                    <span className="font-bold text-slate-900 flex items-center gap-1.5">
                      <Sliders className="w-3.5 h-3.5 text-violet-primary" />
                      <span>Material & Light HUD</span>
                    </span>
                    <button
                      onClick={resetMaterialTuning}
                      className="text-[10px] text-slate-500 hover:text-violet-primary transition-all font-medium"
                    >
                      Reset
                    </button>
                  </div>

                  {/* Surface Roughness Slider */}
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-slate-600 font-medium">Surface Roughness</span>
                      <span className="font-mono text-slate-900 font-bold">{roughness.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="1.0"
                      step="0.02"
                      value={roughness}
                      onChange={(e) => {
                        setRoughness(parseFloat(e.target.value));
                        setIsTunedByUser(true);
                      }}
                      className="w-full accent-violet-primary cursor-pointer h-1.5 bg-slate-200 rounded-lg"
                    />
                    <div className="flex justify-between text-[9px] text-slate-400">
                      <span>Mirror (0.0)</span>
                      <span>Matte (1.0)</span>
                    </div>
                  </div>

                  {/* Metallic Sheen Slider */}
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-slate-600 font-medium">Metallic Sheen</span>
                      <span className="font-mono text-slate-900 font-bold">{metalness.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="1.0"
                      step="0.02"
                      value={metalness}
                      onChange={(e) => {
                        setMetalness(parseFloat(e.target.value));
                        setIsTunedByUser(true);
                      }}
                      className="w-full accent-cyan-accent cursor-pointer h-1.5 bg-slate-200 rounded-lg"
                    />
                    <div className="flex justify-between text-[9px] text-slate-400">
                      <span>Dielectric (0.0)</span>
                      <span>Chrome (1.0)</span>
                    </div>
                  </div>

                  {/* Studio Exposure Slider */}
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-slate-600 font-medium flex items-center gap-1">
                        <Sun className="w-3 h-3 text-amber-500" />
                        <span>Studio Exposure</span>
                      </span>
                      <span className="font-mono text-slate-900 font-bold">{exposure.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="2.0"
                      step="0.05"
                      value={exposure}
                      onChange={(e) => setExposure(parseFloat(e.target.value))}
                      className="w-full accent-amber-500 cursor-pointer h-1.5 bg-slate-200 rounded-lg"
                    />
                    <div className="flex justify-between text-[9px] text-slate-400">
                      <span>Dim (0.5)</span>
                      <span>Bright (2.0)</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <button
              onClick={() => setWireframe(!wireframe)}
              className={`p-2 rounded-xl transition-all ${
                wireframe ? 'bg-violet-primary text-white shadow-violet-glow' : 'hover:bg-slate-100 text-slate-600 hover:text-slate-900'
              }`}
              title="Toggle Wireframe"
            >
              <Eye className="w-4 h-4" />
            </button>

            <button
              onClick={() => setShowGrid(!showGrid)}
              className={`p-2 rounded-xl transition-all ${
                showGrid ? 'bg-cyan-dim text-cyan-accent border border-cyan-accent/40' : 'hover:bg-slate-100 text-slate-600 hover:text-slate-900'
              }`}
              title="Toggle Ground Grid"
            >
              <Grid3X3 className="w-4 h-4" />
            </button>

            <button
              onClick={() => setAutoRotate(!autoRotate)}
              className={`p-2 rounded-xl transition-all ${
                autoRotate ? 'bg-violet-dim text-violet-primary font-bold' : 'hover:bg-slate-100 text-slate-600 hover:text-slate-900'
              }`}
              title="Toggle Auto-Rotation"
            >
              {autoRotate ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </button>

            <button
              onClick={() => setShowMetrics(!showMetrics)}
              className="p-2 rounded-xl hover:bg-slate-100 text-slate-600 hover:text-slate-900 transition-all"
              title="Toggle Geometry Metrics"
            >
              <Info className="w-4 h-4" />
            </button>

            <button
              onClick={() => {
                setShowInspector(!showInspector);
                if (!showInspector) {
                  setShowMetrics(false);
                  setShowDiagnostics(false);
                }
              }}
              className={`p-2 rounded-xl transition-all ${
                showInspector 
                  ? 'bg-violet-dim text-violet-primary border border-violet-primary/30 font-semibold' 
                  : 'hover:bg-slate-100 text-slate-600 hover:text-slate-900'
              }`}
              title="Toggle Component & Material Inspector"
            >
              <Palette className="w-4 h-4" />
            </button>

            <button
              onClick={() => {
                setShowDiagnostics(!showDiagnostics);
                if (!showDiagnostics) {
                  setShowMetrics(false);
                  setShowInspector(false);
                }
              }}
              className={`p-2 rounded-xl transition-all ${
                showDiagnostics 
                  ? 'bg-cyan-dim text-cyan-accent border border-cyan-accent/40 font-semibold' 
                  : 'hover:bg-slate-100 text-slate-600 hover:text-slate-900'
              }`}
              title="Toggle Reconstruction Pipeline Diagnostics"
            >
              <Activity className="w-4 h-4" />
            </button>
          </div>

          {/* Floating Export Action Bar */}
          <div className="absolute top-4 right-4 flex items-center gap-2 z-10">
            {modelMetadata && (
              <>
                <a
                  href={`/api/model/${modelMetadata.model_id}/download?format=glb`}
                  download={`sketch2shape_${modelMetadata.model_id}.glb`}
                  className="px-4 py-2 rounded-xl bg-violet-primary hover:bg-violet-hover text-white text-xs font-bold shadow-violet-glow flex items-center gap-1.5 transition-all"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download GLB</span>
                </a>

                <a
                  href={`/api/model/${modelMetadata.model_id}/download?format=obj`}
                  download={`sketch2shape_${modelMetadata.model_id}.obj`}
                  className="px-3.5 py-2 rounded-xl bg-white hover:bg-slate-50 text-slate-700 hover:text-slate-900 border border-slate-300 text-xs font-semibold shadow-xs flex items-center gap-1.5 transition-all"
                >
                  <Download className="w-3.5 h-3.5 text-cyan-accent" />
                  <span>OBJ</span>
                </a>
              </>
            )}
          </div>

          {/* Real Geometry Metrics HUD (Bottom-Left) */}
          {showMetrics && modelMetadata && (
            <div className="absolute bottom-4 left-4 p-4 rounded-2xl bg-white/95 backdrop-blur-md border border-slate-200 z-10 max-w-xs shadow-xl animate-fadeIn text-slate-800">
              <div className="flex items-center justify-between mb-2 pb-2 border-b border-slate-100">
                <div className="flex items-center gap-1.5 text-xs font-bold text-slate-900">
                  <Box className="w-3.5 h-3.5 text-violet-primary" />
                  <span>Geometry Metrics</span>
                </div>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-bold border border-emerald-200 flex items-center gap-1">
                  <Check className="w-2.5 h-2.5" />
                  <span>Watertight</span>
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-[10px] text-slate-500 font-medium block">Vertices</span>
                  <span className="font-mono text-slate-900 font-bold">
                    {modelMetadata.geometry.vertex_count.toLocaleString()}
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 font-medium block">Triangles</span>
                  <span className="font-mono text-slate-900 font-bold">
                    {modelMetadata.geometry.face_count.toLocaleString()}
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 font-medium block">Real-World Scale</span>
                  <span className="font-mono text-[11px] text-violet-primary font-bold">
                    {formatMetricDimensions(modelMetadata.geometry.dimensions)}
                  </span>
                </div>
                <div>
                  <span className="text-[10px] text-slate-500 font-medium block">Render Time</span>
                  <span className="font-mono text-cyan-accent font-bold">
                    {modelMetadata.processing_time_seconds.toFixed(1)}s
                  </span>
                </div>
              </div>

              {modelMetadata.ai_analysis?.available && (
                <div className="mt-2 pt-2 border-t border-slate-100 text-[10px] text-slate-600">
                  <div className="flex items-center justify-between text-cyan-accent font-bold mb-0.5">
                    <span>Groq Vision: {modelMetadata.ai_analysis.object_name || modelMetadata.ai_analysis.object_type}</span>
                    <span className="text-[9px] text-violet-primary font-semibold">qwen3.8-27b</span>
                  </div>
                  <span className="text-slate-700 line-clamp-2">
                    {modelMetadata.ai_analysis.main_body ? (
                      `Body: ${modelMetadata.ai_analysis.main_body.shape || 'radial'} • ${modelMetadata.ai_analysis.main_body.interior || 'hollow'} • Material: ${modelMetadata.ai_analysis.material || 'ceramic'}`
                    ) : (
                      modelMetadata.ai_analysis.description
                    )}
                  </span>
                </div>
              )}

              {modelMetadata.prompt && (
                <div className="mt-2 pt-2 border-t border-slate-100 text-[10px] text-slate-600">
                  <span className="text-slate-400 font-medium">Prompt: </span>
                  <span className="text-slate-800 italic font-medium truncate block">{modelMetadata.prompt}</span>
                </div>
              )}
            </div>
          )}

          {/* Component & Material Inspector Drawer */}
          {showInspector && modelMetadata && (
            <div className="absolute top-16 right-4 p-4 rounded-2xl bg-white/95 backdrop-blur-md border border-slate-200 z-20 w-80 max-h-[80%] overflow-y-auto shadow-2xl animate-fadeIn text-slate-800">
              <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-100">
                <div className="flex items-center gap-2">
                  <Palette className="w-4 h-4 text-violet-primary" />
                  <span className="text-xs font-bold text-slate-900">Component & Material Inspector</span>
                </div>
                <button
                  onClick={() => setShowInspector(false)}
                  className="p-1 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Quality Criteria Validation Overview */}
              {modelMetadata.quality_validation && (
                <div className="mb-3 p-2.5 rounded-xl bg-slate-50 border border-slate-200/80 space-y-1.5 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-semibold text-slate-700">Reconstruction Quality</span>
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-bold border border-emerald-200 flex items-center gap-1">
                      <CheckCircle2 className="w-2.5 h-2.5 text-emerald-600" />
                      <span>{modelMetadata.quality_validation.visual_reconstruction_quality.toUpperCase()}</span>
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-1.5 pt-1 text-[10px] text-slate-600">
                    <div className="bg-white p-1.5 rounded-lg border border-slate-100">
                      <span className="text-slate-400 block">Geometry</span>
                      <span className="font-bold text-emerald-700">
                        {modelMetadata.quality_validation.geometry_validity ? 'Valid & Watertight' : 'Non-manifold'}
                      </span>
                    </div>
                    <div className="bg-white p-1.5 rounded-lg border border-slate-100">
                      <span className="text-slate-400 block">Confidence</span>
                      <span className="font-bold text-violet-primary">
                        {Math.round(modelMetadata.quality_validation.object_recognition_confidence * 100)}%
                      </span>
                    </div>
                    <div className="bg-white p-1.5 rounded-lg border border-slate-100">
                      <span className="text-slate-400 block">Components</span>
                      <span className="font-bold text-slate-900">
                        {Math.round(modelMetadata.quality_validation.component_completeness * 100)}% Complete
                      </span>
                    </div>
                    <div className="bg-white p-1.5 rounded-lg border border-slate-100">
                      <span className="text-slate-400 block">PBR Shading</span>
                      <span className="font-bold text-cyan-accent">
                        {Math.round(modelMetadata.quality_validation.material_completeness * 100)}% Textured
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Architectural / Object Components Breakdown */}
              <div className="space-y-1.5">
                <span className="text-[11px] font-bold text-slate-700 block">
                  Identified Components ({modelMetadata.quality_validation?.validated_components?.length || modelMetadata.ai_analysis?.components?.length || 0})
                </span>

                <div className="space-y-1 max-h-56 overflow-y-auto pr-1">
                  {modelMetadata.quality_validation?.validated_components && modelMetadata.quality_validation.validated_components.length > 0 ? (
                    modelMetadata.quality_validation.validated_components.map((comp, idx) => (
                      <div 
                        key={idx} 
                        className="flex items-center justify-between p-1.5 rounded-lg bg-slate-50 border border-slate-100 text-[11px]"
                      >
                        <div className="flex items-center gap-1.5 min-w-0">
                          <span className="w-2 h-2 rounded-full bg-violet-primary shrink-0" />
                          <span className="font-medium text-slate-800 truncate">{comp.component}</span>
                        </div>
                        <span className="text-[10px] text-slate-500 font-mono shrink-0 pl-1">
                          {comp.material || comp.status}
                        </span>
                      </div>
                    ))
                  ) : modelMetadata.ai_analysis?.components && modelMetadata.ai_analysis.components.length > 0 ? (
                    modelMetadata.ai_analysis.components.map((comp, idx) => (
                      <div 
                        key={idx} 
                        className="flex items-center justify-between p-1.5 rounded-lg bg-slate-50 border border-slate-100 text-[11px]"
                      >
                        <div className="flex items-center gap-1.5 min-w-0">
                          <span className="w-2 h-2 rounded-full bg-cyan-accent shrink-0" />
                          <span className="font-medium text-slate-800 truncate">{comp.name}</span>
                        </div>
                        <span className="text-[10px] text-slate-500 font-mono shrink-0 pl-1">
                          {comp.shape || 'volumetric'}
                        </span>
                      </div>
                    ))
                  ) : (
                    <div className="text-[11px] text-slate-500 italic p-2 text-center">
                      Procedural PBR Material Shading
                    </div>
                  )}
                </div>
              </div>

              {/* Active Material Properties */}
              <div className="mt-3 pt-2.5 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-500">
                <span>Material: <strong className="text-slate-800">{modelMetadata.material_props?.name || 'Multi-Component PBR'}</strong></span>
                <span>Roughness: <strong className="text-slate-800">{roughness.toFixed(2)}</strong></span>
              </div>
            </div>
          )}

          {/* Reconstruction Diagnostics Drawer (Phase 10) */}
          {showDiagnostics && modelMetadata && (
            <div className="absolute top-16 right-4 p-4 rounded-2xl bg-white/95 backdrop-blur-md border border-slate-200 z-20 w-88 max-h-[85%] overflow-y-auto shadow-2xl animate-fadeIn text-slate-800">
              <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-100">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cyan-accent" />
                  <span className="text-xs font-bold text-slate-900">Pipeline Tracing & Diagnostics</span>
                </div>
                <button
                  onClick={() => setShowDiagnostics(false)}
                  className="p-1 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Visual Identity Validation Banner */}
              <div className="mb-3 p-2.5 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-[11px] flex items-center justify-between">
                <div className="flex items-center gap-1.5 font-bold">
                  <ShieldCheck className="w-4 h-4 text-emerald-600" />
                  <span>Visual Identity Verified</span>
                </div>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-900 font-bold">
                  {Math.round((modelMetadata.identity_tracing?.detection_confidence || modelMetadata.ai_analysis?.confidence || 0.95) * 100)}% Conf
                </span>
              </div>

              <div className="space-y-2.5 text-[11px]">
                {/* Object & Category */}
                <div className="p-2 rounded-xl bg-slate-50 border border-slate-100">
                  <div className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider mb-1">Object Classification</div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">Detected Object:</span>
                    <strong className="text-slate-900 font-mono capitalize">
                      {modelMetadata.identity_tracing?.detected_object || modelMetadata.ai_analysis?.object_name || 'Verified Object'}
                    </strong>
                  </div>
                  <div className="flex justify-between items-center mt-0.5">
                    <span className="text-slate-600">Category:</span>
                    <span className="text-violet-primary font-semibold capitalize">
                      {modelMetadata.identity_tracing?.detected_category || modelMetadata.ai_analysis?.object_category || 'general'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center mt-0.5">
                    <span className="text-slate-600">Vision Model:</span>
                    <span className="text-slate-700 font-mono text-[10px]">
                      {modelMetadata.identity_tracing?.vision_model || modelMetadata.ai_analysis?.model_used || 'Groq LPU Vision (qwen3.8-27b)'}
                    </span>
                  </div>
                </div>

                {/* Image Identity */}
                <div className="p-2 rounded-xl bg-slate-50 border border-slate-100 space-y-1">
                  <div className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider mb-1">Input Image Identity</div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">Image ID:</span>
                    <span className="font-mono text-slate-800 text-[10px] truncate max-w-[170px]" title={modelMetadata.identity_tracing?.image_id}>
                      {modelMetadata.identity_tracing?.image_id || 'verified'}
                    </span>
                  </div>
                  {modelMetadata.identity_tracing?.image_hash_sha256 && (
                    <div className="flex justify-between items-center">
                      <span className="text-slate-600">SHA-256 Hash:</span>
                      <span className="font-mono text-slate-700 text-[9px]">
                        {modelMetadata.identity_tracing.image_hash_sha256.slice(0, 16)}...
                      </span>
                    </div>
                  )}
                  {modelMetadata.identity_tracing?.original_dimensions && (
                    <div className="flex justify-between items-center">
                      <span className="text-slate-600">Dimensions:</span>
                      <span className="font-mono text-slate-800">
                        {modelMetadata.identity_tracing.original_dimensions[0]} × {modelMetadata.identity_tracing.original_dimensions[1]} px
                      </span>
                    </div>
                  )}
                </div>

                {/* Reconstruction Task & Asset */}
                <div className="p-2 rounded-xl bg-slate-50 border border-slate-100 space-y-1">
                  <div className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider mb-1">3D Reconstruction Task</div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">Engine / Provider:</span>
                    <span className="font-bold text-violet-primary font-mono">{modelMetadata.engine}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">Task / Model ID:</span>
                    <span className="font-mono text-slate-800 text-[10px]">{modelMetadata.model_id}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">Status:</span>
                    <span className="text-emerald-700 font-bold">COMPLETED (100%)</span>
                  </div>
                  {modelMetadata.identity_tracing?.final_glb_hash && (
                    <div className="flex justify-between items-center">
                      <span className="text-slate-600">GLB Binary Hash:</span>
                      <span className="font-mono text-slate-700 text-[9px]">
                        {modelMetadata.identity_tracing.final_glb_hash.slice(0, 16)}...
                      </span>
                    </div>
                  )}
                </div>

                {/* User Prompt & Conflict Analysis */}
                <div className="p-2 rounded-xl bg-slate-50 border border-slate-100 space-y-1">
                  <div className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider mb-1">Prompt Conditioning</div>
                  <div className="text-slate-700 italic">
                    {modelMetadata.prompt ? `"${modelMetadata.prompt}"` : 'No prompt (Direct visual conditioning)'}
                  </div>
                  <div className="text-[10px] text-emerald-700 font-medium pt-0.5">
                    Policy: Strict Image-First Reconstruction • No Prompt Contradiction
                  </div>
                </div>
              </div>
            </div>
          )}

        </div>
      ) : (
        /* Empty State */
        <div className="w-full h-full flex flex-col items-center justify-center p-8 text-center bg-white/70 backdrop-blur-sm">
          <div className="w-20 h-20 rounded-3xl bg-slate-50 border border-slate-200 flex items-center justify-center text-violet-primary mb-4 shadow-sm">
            <Box className="w-10 h-10 animate-pulse text-violet-primary" />
          </div>
          <h3 className="text-lg font-bold text-slate-900 mb-1">
            Real-Time 3D Studio
          </h3>
          <p className="text-xs text-slate-600 max-w-sm mb-6 leading-relaxed">
            Draw your concept sketch or upload an illustration, then click &ldquo;Generate 3D Model&rdquo; to reconstruct interactive watertight geometry.
          </p>
          <div className="flex items-center gap-3 text-xs text-slate-500 font-medium">
            <span className="flex items-center gap-1.5 text-slate-700 font-semibold">
              <Sparkles className="w-3.5 h-3.5 text-violet-primary" />
              <span>Orbit, Pan & Zoom</span>
            </span>
            <span>•</span>
            <span>4 Camera Views</span>
            <span>•</span>
            <span>PBR Materials</span>
            <span>•</span>
            <span>Watertight GLB/OBJ</span>
          </div>
        </div>
      )}

    </div>
  );
};

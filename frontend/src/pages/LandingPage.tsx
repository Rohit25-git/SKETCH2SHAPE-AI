import React from 'react';
import { 
  Sparkles, 
  ArrowRight, 
  Box, 
  Zap, 
  ShieldCheck, 
  Download, 
  Eye, 
  Sliders,
  CheckCircle2
} from 'lucide-react';
import { Showcase3D } from '../components/landing/Showcase3D';

interface LandingPageProps {
  onStartCreating: () => void;
  onExploreFeatures: () => void;
}

const FEATURES = [
  {
    icon: Sparkles,
    title: "Sketch to True Geometry",
    desc: "Draw freehand or upload 2D artwork. Our volumetric Signed Distance Field pipeline synthesizes real watertight manifold 3D meshes.",
  },
  {
    icon: Eye,
    title: "Interactive Studio Inspection",
    desc: "Inspect geometry in a real-time WebGL/Three.js studio with orbit controls, wireframe analysis, and ground grid alignment.",
  },
  {
    icon: Download,
    title: "Production GLB & OBJ Export",
    desc: "Download verified binary GLB and wavefront OBJ models directly compatible with Blender, Unreal Engine, Unity, and 3D printers.",
  },
  {
    icon: Zap,
    title: "Hardware-Adaptive Speed",
    desc: "Runs smoothly on local machines with optimized Marching Cubes, or scales to Cloud Neural AI providers when API keys are configured.",
  },
  {
    icon: ShieldCheck,
    title: "Watertight Manifold Meshes",
    desc: "Zero fake billboard planes or flat texture tricks. Every asset is mathematically validated for manifoldness and non-zero volume.",
  },
  {
    icon: Sliders,
    title: "Granular Topology Control",
    desc: "Fine-tune voxel resolution, Laplacian smoothing passes, and volumetric thickness to craft exact geometry profiles.",
  },
];

const WORKFLOW_STEPS = [
  {
    step: "01",
    title: "Illustrate or Upload",
    desc: "Draw directly on our responsive 2D canvas with brush, eraser, and color tools, or drop in any PNG/JPG sketch.",
  },
  {
    step: "02",
    title: "OpenCV Preprocessing",
    desc: "Automatic noise reduction, stroke enhancement, and contour extraction isolate clean silhouette geometries.",
  },
  {
    step: "03",
    title: "Volumetric Reconstruction",
    desc: "Mathematical distance transform and 3D Marching Cubes generate watertight manifold polygonal geometry.",
  },
  {
    step: "04",
    title: "Inspect & Export",
    desc: "Orbit in 3D, verify real vertex/face metrics, and export binary GLB or OBJ assets with one click.",
  },
];

export const LandingPage: React.FC<LandingPageProps> = ({
  onStartCreating,
  onExploreFeatures,
}) => {
  return (
    <div className="flex flex-col gap-24 py-8 px-4 lg:px-8 max-w-7xl mx-auto">
      
      {/* Hero Section */}
      <section className="relative pt-6 pb-12 flex flex-col lg:flex-row items-center gap-12">
        
        {/* Left Column: Copy & Actions */}
        <div className="flex-1 flex flex-col gap-6 text-left">
          
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white border border-slate-200 text-xs text-slate-600 w-fit shadow-xs">
            <span className="w-2 h-2 rounded-full bg-violet-primary animate-pulse" />
            <span className="text-slate-900 font-bold">SKETCH2SHAPE AI Studio</span>
            <span className="text-slate-300">•</span>
            <span className="text-cyan-accent font-semibold">Watertight 3D Pipeline</span>
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-slate-900 leading-[1.1]">
            From a simple sketch to a{' '}
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-violet-primary via-indigo-600 to-cyan-accent">
              dimensional reality.
            </span>
          </h1>

          <p className="text-base sm:text-lg text-slate-600 max-w-xl leading-relaxed font-medium">
            A premium AI-powered creative studio where you can draw or upload a 2D sketch, describe your vision, generate genuine 3D manifold geometry, inspect it interactively, and export standard GLB/OBJ assets.
          </p>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            <button
              onClick={onStartCreating}
              className="px-6 py-3.5 rounded-2xl bg-gradient-to-r from-violet-primary to-violet-hover hover:from-violet-hover hover:to-indigo-700 text-white text-sm font-bold shadow-violet-glow transition-all flex items-center gap-2.5 group cursor-pointer"
            >
              <span>Start Creating Free</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>

            <button
              onClick={onExploreFeatures}
              className="px-6 py-3.5 rounded-2xl bg-white hover:bg-slate-50 border border-slate-200 text-slate-800 text-sm font-bold transition-all shadow-xs cursor-pointer"
            >
              Explore Features
            </button>
          </div>

          <div className="flex items-center gap-6 pt-4 text-xs text-slate-500 font-semibold">
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>Genuine 3D Geometry</span>
            </div>
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>No Fake Billboard Planes</span>
            </div>
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>GLB & OBJ Export</span>
            </div>
          </div>

        </div>

        {/* Right Column: Interactive 3D Showcase */}
        <div className="flex-1 w-full max-w-lg lg:max-w-none">
          <Showcase3D />
        </div>

      </section>

      {/* How It Works Section */}
      <section className="flex flex-col gap-12 text-center pt-8">
        <div>
          <span className="text-xs uppercase tracking-widest text-violet-primary font-bold">
            Workflow Architecture
          </span>
          <h2 className="text-3xl font-extrabold text-slate-900 mt-2">
            How SKETCH2SHAPE Operates
          </h2>
          <p className="text-sm text-slate-600 max-w-lg mx-auto mt-2 font-medium">
            Every step is backed by transparent, deterministic geometric computing and neural processing.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {WORKFLOW_STEPS.map((s) => (
            <div
              key={s.step}
              className="p-6 rounded-3xl bg-white border border-slate-200 text-left flex flex-col gap-3 relative group hover:border-violet-primary/50 transition-all shadow-studio-card"
            >
              <span className="text-2xl font-black text-slate-300 font-mono group-hover:text-violet-primary transition-colors">
                {s.step}
              </span>
              <h3 className="text-base font-bold text-slate-900 group-hover:text-violet-primary transition-colors">
                {s.title}
              </h3>
              <p className="text-xs text-slate-600 leading-relaxed font-medium">
                {s.desc}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Features Grid */}
      <section id="features" className="flex flex-col gap-12 text-center pt-8">
        <div>
          <span className="text-xs uppercase tracking-widest text-cyan-accent font-bold">
            Studio Capabilities
          </span>
          <h2 className="text-3xl font-extrabold text-slate-900 mt-2">
            Engineered for Serious Creators
          </h2>
          <p className="text-sm text-slate-600 max-w-lg mx-auto mt-2 font-medium">
            Built from the ground up for high fidelity 3D asset generation and real-time interaction.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {FEATURES.map((f, i) => {
            const Icon = f.icon;
            return (
              <div
                key={i}
                className="p-6 rounded-3xl bg-white border border-slate-200 text-left flex flex-col gap-3 group hover:border-violet-primary/50 transition-all shadow-studio-card"
              >
                <div className="w-12 h-12 rounded-2xl bg-slate-50 border border-slate-200 flex items-center justify-center text-violet-primary group-hover:scale-110 group-hover:bg-violet-dim transition-all">
                  <Icon className="w-6 h-6" />
                </div>
                <h3 className="text-base font-bold text-slate-900">
                  {f.title}
                </h3>
                <p className="text-xs text-slate-600 leading-relaxed font-medium">
                  {f.desc}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      {/* CTA Banner */}
      <section className="relative rounded-3xl p-8 lg:p-14 bg-gradient-to-r from-violet-dim via-white to-cyan-dim border border-slate-200 flex flex-col lg:flex-row items-center justify-between gap-8 shadow-studio-card text-left overflow-hidden">
        <div className="flex flex-col gap-2 max-w-xl z-10">
          <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900">
            Ready to bring your sketches to life?
          </h2>
          <p className="text-xs sm:text-sm text-slate-600 font-medium">
            Launch the SKETCH2SHAPE studio now to draw, configure directives, and download watertight 3D models.
          </p>
        </div>

        <button
          onClick={onStartCreating}
          className="px-8 py-4 rounded-2xl bg-violet-primary hover:bg-violet-hover text-white text-sm font-bold shadow-violet-glow transition-all flex items-center gap-2 group shrink-0 z-10 cursor-pointer"
        >
          <span>Open Creative Studio</span>
          <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
        </button>
      </section>

      {/* Footer */}
      <footer className="pt-12 pb-6 border-t border-slate-200 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500 font-medium">
        <div className="flex items-center gap-2">
          <Box className="w-4 h-4 text-violet-primary" />
          <span className="font-bold text-slate-900">SKETCH2SHAPE AI</span>
          <span>&mdash; &ldquo;From a simple sketch to a dimensional reality.&rdquo;</span>
        </div>
        <p>&copy; {new Date().getFullYear()} SKETCH2SHAPE AI. All rights reserved.</p>
      </footer>

    </div>
  );
};

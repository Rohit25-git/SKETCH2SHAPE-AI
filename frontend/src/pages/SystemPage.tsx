import React from 'react';
import { 
  Cpu, 
  HardDrive, 
  Activity, 
  Key, 
  Layers, 
  CheckCircle2, 
  Server,
  Terminal
} from 'lucide-react';
import type { SystemHealth } from '../types';

interface SystemPageProps {
  health: SystemHealth | null;
}

export const SystemPage: React.FC<SystemPageProps> = ({ health }) => {
  return (
    <div className="max-w-5xl mx-auto px-4 lg:px-8 py-8 flex flex-col gap-8 text-left">
      
      {/* Header */}
      <div className="pb-6 border-b border-slate-200">
        <h1 className="text-2xl font-extrabold text-slate-900 flex items-center gap-2.5">
          <Cpu className="w-6 h-6 text-violet-primary" />
          <span>System & Hardware Transparency</span>
        </h1>
        <p className="text-xs text-slate-500 font-medium mt-1">
          Real-time hardware inspection, model pipeline execution status, and API configurations.
        </p>
      </div>

      {/* Hardware Metrics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        
        {/* Operating System */}
        <div className="p-5 rounded-2xl bg-white border border-slate-200 shadow-sm flex flex-col gap-2">
          <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
            <span>Operating System</span>
            <Server className="w-4 h-4 text-cyan-accent" />
          </div>
          <span className="text-sm font-bold text-slate-900 truncate">
            {health?.hardware.os || 'Windows 64-bit'}
          </span>
          <span className="text-[10px] text-slate-400 font-medium">Platform Architecture</span>
        </div>

        {/* CPU Cores */}
        <div className="p-5 rounded-2xl bg-white border border-slate-200 shadow-sm flex flex-col gap-2">
          <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
            <span>Logical Cores</span>
            <Activity className="w-4 h-4 text-emerald-500" />
          </div>
          <span className="text-xl font-bold text-slate-900 font-mono">
            {health?.hardware.cpu_count || 4} Threads
          </span>
          <span className="text-[10px] text-emerald-600 font-semibold">Parallel SDF Computing</span>
        </div>

        {/* System RAM */}
        <div className="p-5 rounded-2xl bg-white border border-slate-200 shadow-sm flex flex-col gap-2">
          <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
            <span>System Memory</span>
            <HardDrive className="w-4 h-4 text-violet-primary" />
          </div>
          <span className="text-xl font-bold text-slate-900 font-mono">
            {health?.hardware.ram_available_gb.toFixed(1)} / {health?.hardware.ram_total_gb.toFixed(1)} GB
          </span>
          <span className="text-[10px] text-slate-500 font-medium">Available / Total</span>
        </div>

        {/* GPU / CUDA Status */}
        <div className="p-5 rounded-2xl bg-white border border-slate-200 shadow-sm flex flex-col gap-2">
          <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
            <span>NVIDIA CUDA</span>
            <Cpu className="w-4 h-4 text-amber-500" />
          </div>
          <span className="text-sm font-bold text-slate-900">
            {health?.hardware.cuda_available ? health.hardware.gpu_name : 'Adaptive CPU Engine'}
          </span>
          <span className="text-[10px] text-cyan-accent font-semibold">
            {health?.hardware.cuda_available ? 'CUDA Acceleration Active' : 'Optimized Vectorized CPU'}
          </span>
        </div>

      </div>

      {/* Reconstruction Engines Matrix */}
      <div className="p-6 rounded-3xl bg-white border border-slate-200 shadow-sm flex flex-col gap-5">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <Layers className="w-4 h-4 text-violet-primary" />
          <span>Active Reconstruction Engines</span>
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          
          {/* Procedural Engine */}
          <div className="p-4 rounded-2xl bg-slate-50 border border-violet-primary/40 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-900 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500" />
                <span>Procedural Manifold SDF Engine</span>
              </span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 font-bold border border-emerald-200">
                ACTIVE
              </span>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed font-medium">
              Deterministic computer vision & geometric reconstruction. Computes Euclidean Distance Transforms, Signed Distance Fields, and Marching Cubes isosurface extraction to generate watertight 3D manifold meshes locally.
            </p>
            <div className="text-[11px] text-slate-500 flex items-center gap-2 pt-1 border-t border-slate-200 font-medium">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              <span>Offline Ready • Watertight Verified • CPU/GPU</span>
            </div>
          </div>

          {/* Generative AI Engine */}
          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-900 flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${health?.cloud_api_configured.tripo ? 'bg-cyan-accent' : 'bg-slate-400'}`} />
                <span>Generative AI Engine (Tripo3D API)</span>
              </span>
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold border ${
                health?.cloud_api_configured.tripo
                  ? 'bg-cyan-dim text-cyan-accent border-cyan-accent/30'
                  : 'bg-slate-200 text-slate-500 border-slate-300'
              }`}>
                {health?.cloud_api_configured.tripo ? 'CONFIGURED' : 'OPTIONAL (API KEY)'}
              </span>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed font-medium">
              True generative deep neural reconstruction (Large Reconstruction Model / Tripo LRM). Ingests 2D sketches and optional natural-language guidance to generate detailed multi-view 3D models.
            </p>
            <div className="text-[11px] text-slate-500 flex items-center gap-2 pt-1 border-t border-slate-200 font-medium">
              <Key className="w-3.5 h-3.5 text-amber-500" />
              <span>Requires TRIPO_API_KEY in .env</span>
            </div>
          </div>

        </div>
      </div>

      {/* Quick Startup Reference */}
      <div className="p-6 rounded-3xl bg-white border border-slate-200 shadow-sm flex flex-col gap-3">
        <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <Terminal className="w-4 h-4 text-cyan-accent" />
          <span>Execution Commands</span>
        </h2>
        <div className="bg-slate-900 p-4 rounded-2xl border border-slate-800 font-mono text-xs text-slate-300 flex flex-col gap-2">
          <div>
            <span className="text-slate-400 font-medium"># Start Backend:</span>
            <div className="text-cyan-accent">.\venv\Scripts\uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload</div>
          </div>
          <div className="pt-2 border-t border-slate-800">
            <span className="text-slate-400 font-medium"># Start Frontend:</span>
            <div className="text-violet-glow">cd frontend && npm run dev</div>
          </div>
          <div className="pt-2 border-t border-slate-800">
            <span className="text-slate-400 font-medium"># Run Pytest Suite:</span>
            <div className="text-emerald-400">.\venv\Scripts\pytest backend/tests/ -v</div>
          </div>
        </div>
      </div>

    </div>
  );
};

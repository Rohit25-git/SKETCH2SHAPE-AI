import React, { useState } from 'react';
import { Sparkles, Sliders, Cpu } from 'lucide-react';
import type { SystemHealth } from '../../types';

interface GenerationControlsProps {
  prompt: string;
  setPrompt: (p: string) => void;
  engine: string;
  setEngine: (e: string) => void;
  resolution: string;
  setResolution: (r: string) => void;
  smoothness: number;
  setSmoothness: (s: number) => void;
  thickness: number;
  setThickness: (t: number) => void;
  onGenerate: () => void;
  isGenerating: boolean;
  hasInput: boolean;
  health: SystemHealth | null;
}

const SAMPLE_PROMPTS = [
  "Futuristic ergonomic armchair with metallic frame",
  "Minimalist Scandinavian ceramic vase",
  "Cyberpunk sci-fi drone with sleek thrusters",
  "Classic espresso coffee mug with rounded handle"
];

export const GenerationControls: React.FC<GenerationControlsProps> = ({
  prompt,
  setPrompt,
  engine,
  setEngine,
  resolution,
  setResolution,
  smoothness,
  setSmoothness,
  thickness,
  setThickness,
  onGenerate,
  isGenerating,
  hasInput,
  health,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  return (
    <div className="flex flex-col gap-4 p-5 rounded-3xl bg-studio-surface border border-studio-border shadow-studio-card">
      
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-violet-primary" />
          <h3 className="text-sm font-bold text-slate-900">Generation Directives</h3>
        </div>
        <div className="flex items-center gap-2">
          {health?.groq_vision_configured && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-dim text-cyan-accent border border-cyan-accent/30 font-semibold flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-accent animate-ping" />
              <span>Groq Vision Active</span>
            </span>
          )}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs font-medium text-studio-muted hover:text-studio-text flex items-center gap-1 transition-colors"
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>{showAdvanced ? 'Hide Settings' : 'Settings'}</span>
          </button>
        </div>
      </div>

      {/* Natural Language Prompt */}
      <div className="flex flex-col gap-2">
        <label className="text-xs font-semibold text-slate-700 flex items-center justify-between">
          <span>Natural-Language Prompt (Optional)</span>
          <span className="text-[10px] text-slate-400 font-normal">Refines mesh semantics</span>
        </label>
        <textarea
          rows={2}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g. A sleek wooden dining chair with curved backrest..."
          className="w-full px-3.5 py-2.5 rounded-xl bg-slate-50 border border-slate-200 focus:border-violet-primary focus:bg-white focus:outline-none text-xs text-slate-900 placeholder-slate-400 font-medium transition-all resize-none shadow-2xs"
        />

        {/* Suggestion Chips */}
        <div className="flex flex-wrap gap-1.5 mt-1">
          {SAMPLE_PROMPTS.map((sample, i) => (
            <button
              key={i}
              type="button"
              onClick={() => setPrompt(sample)}
              className="text-[10px] px-2.5 py-1 rounded-lg bg-slate-50 hover:bg-slate-100 text-slate-600 hover:text-slate-900 border border-slate-200 transition-all text-left truncate max-w-[220px] font-medium"
              title={sample}
            >
              {sample}
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Settings Drawer */}
      {showAdvanced && (
        <div className="flex flex-col gap-3.5 pt-3 border-t border-slate-100 animate-fadeIn">
          
          {/* Engine Selector */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-700 font-semibold flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-cyan-accent" />
                <span>Reconstruction Engine</span>
              </span>
              <span className="text-[10px] text-slate-400">
                {engine === 'local_manifold' ? 'Procedural SDF + Marching Cubes' : 'Generative Neural AI (Tripo LRM)'}
              </span>
            </div>
            
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setEngine('local_manifold')}
                className={`p-2.5 rounded-xl border text-left text-xs transition-all ${
                  engine === 'local_manifold'
                    ? 'border-violet-primary bg-violet-dim/50 text-slate-900 ring-1 ring-violet-primary/30'
                    : 'border-slate-200 bg-slate-50 text-slate-600 hover:text-slate-900'
                }`}
              >
                <div className="font-bold flex items-center gap-1.5 text-slate-900">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  <span>Procedural SDF</span>
                </div>
                <p className="text-[10px] text-slate-500 mt-0.5">Watertight Marching Cubes (Offline)</p>
              </button>

              <button
                type="button"
                onClick={() => setEngine('cloud_tripo')}
                disabled={!health?.cloud_api_configured.tripo}
                className={`p-2.5 rounded-xl border text-left text-xs transition-all ${
                  engine === 'cloud_tripo'
                    ? 'border-cyan-accent bg-cyan-dim/40 text-slate-900 ring-1 ring-cyan-accent/30'
                    : 'border-slate-200 bg-slate-50 text-slate-600 hover:text-slate-900'
                } ${!health?.cloud_api_configured.tripo ? 'opacity-40 cursor-not-allowed' : ''}`}
                title={!health?.cloud_api_configured.tripo ? 'Configure TRIPO_API_KEY in .env' : 'Cloud Tripo API'}
              >
                <div className="font-bold flex items-center gap-1.5 text-slate-900">
                  <span className={`w-1.5 h-1.5 rounded-full ${health?.cloud_api_configured.tripo ? 'bg-cyan-accent' : 'bg-slate-400'}`} />
                  <span>Generative AI (Tripo)</span>
                </div>
                <p className="text-[10px] text-slate-500 mt-0.5">
                  {health?.cloud_api_configured.tripo ? 'Tripo3D API Active' : 'API Key Required'}
                </p>
              </button>
            </div>
          </div>

          {/* Mesh Resolution */}
          <div className="flex flex-col gap-1.5">
            <span className="text-xs text-slate-700 font-semibold">Voxel Resolution</span>
            <div className="grid grid-cols-3 gap-2">
              {['fast', 'standard', 'high'].map((res) => (
                <button
                  key={res}
                  type="button"
                  onClick={() => setResolution(res)}
                  className={`py-1.5 rounded-xl text-xs font-semibold border capitalize transition-all ${
                    resolution === res
                      ? 'border-violet-primary bg-violet-primary text-white shadow-xs'
                      : 'border-slate-200 bg-slate-50 text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {res}
                </button>
              ))}
            </div>
          </div>

          {/* Smoothness Slider */}
          <div className="flex flex-col gap-1">
            <div className="flex justify-between text-xs">
              <span className="text-slate-600 font-medium">Laplacian Smoothing</span>
              <span className="text-slate-900 font-mono font-bold">{smoothness} passes</span>
            </div>
            <input
              type="range"
              min="1"
              max="6"
              value={smoothness}
              onChange={(e) => setSmoothness(Number(e.target.value))}
              className="accent-violet-primary cursor-pointer w-full"
            />
          </div>

          {/* Thickness Factor */}
          <div className="flex flex-col gap-1">
            <div className="flex justify-between text-xs">
              <span className="text-slate-600 font-medium">Volumetric Thickness</span>
              <span className="text-slate-900 font-mono font-bold">{thickness.toFixed(1)}x</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.1"
              value={thickness}
              onChange={(e) => setThickness(Number(e.target.value))}
              className="accent-cyan-accent cursor-pointer w-full"
            />
          </div>

        </div>
      )}

      {/* Primary Action Button */}
      <button
        type="button"
        onClick={onGenerate}
        disabled={!hasInput || isGenerating}
        className={`w-full py-3.5 px-6 rounded-2xl font-bold text-sm transition-all duration-300 flex items-center justify-center gap-2.5 shadow-md ${
          hasInput && !isGenerating
            ? 'bg-gradient-to-r from-violet-primary via-indigo-600 to-cyan-accent hover:opacity-95 hover:scale-[1.01] text-white shadow-violet-glow cursor-pointer'
            : 'bg-slate-200 text-slate-400 cursor-not-allowed'
        }`}
      >
        <Sparkles className={`w-4 h-4 ${isGenerating ? 'animate-spin' : ''}`} />
        <span>{isGenerating ? 'Reconstructing 3D Mesh...' : 'Generate 3D Model'}</span>
      </button>

      {!hasInput && (
        <p className="text-[11px] text-center text-slate-400 font-medium">
          Please draw a sketch on the canvas or upload an image to begin.
        </p>
      )}

    </div>
  );
};

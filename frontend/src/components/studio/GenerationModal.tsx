import React from 'react';
import { 
  CheckCircle2, 
  Loader2, 
  AlertCircle, 
  Clock, 
  RotateCcw, 
  Box, 
  ShieldCheck
} from 'lucide-react';
import type { JobStatus } from '../../types';

interface GenerationModalProps {
  status: JobStatus | null;
  isOpen: boolean;
  onRetry: () => void;
  onClose: () => void;
}

const STAGES_LIST = [
  { key: 'input_validation', label: '1. Input Validation', desc: 'Validating image resolution, aspect ratio & sketch integrity' },
  { key: 'image_preprocessing', label: '2. Image Preprocessing', desc: 'OpenCV stroke enhancement & edge detection' },
  { key: 'background_preparation', label: '3. Background Isolation', desc: 'Alpha channel segmentation & foreground mask' },
  { key: 'reconstruction_initialization', label: '4. Reconstruction Setup', desc: '3D coordinate normalization & distance transform' },
  { key: '3d_geometry_generation', label: '5. Geometry Generation', desc: 'Volumetric Signed Distance Field & Marching Cubes' },
  { key: 'mesh_processing', label: '6. Mesh Processing', desc: 'Laplacian smoothing & vertex normal orientation' },
  { key: 'output_validation', label: '7. Output Validation', desc: 'Watertight manifold check & metric boundary calculation' },
  { key: 'model_ready', label: '8. Model Ready', desc: 'GLB & OBJ assets compiled for real-time inspection' },
];

export const GenerationModal: React.FC<GenerationModalProps> = ({
  status,
  isOpen,
  onRetry,
  onClose,
}) => {
  if (!isOpen || !status) return null;

  const isFailed = status.status === 'failed';
  const isComplete = status.status === 'completed';

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-xs animate-fadeIn">
      <div className="w-full max-w-lg bg-white border border-slate-200 rounded-3xl p-6 shadow-2xl flex flex-col gap-5">
        
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-100 pb-4">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-2xl flex items-center justify-center ${
              isFailed
                ? 'bg-red-50 text-red-500'
                : isComplete
                ? 'bg-emerald-50 text-emerald-600'
                : 'bg-violet-dim text-violet-primary'
            }`}>
              {isFailed ? (
                <AlertCircle className="w-5 h-5" />
              ) : isComplete ? (
                <ShieldCheck className="w-5 h-5" />
              ) : (
                <Box className="w-5 h-5 animate-bounce" />
              )}
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-900">
                {isFailed
                  ? 'Reconstruction Stopped'
                  : isComplete
                  ? '3D Geometry Generated'
                  : 'Synthesizing 3D Asset'}
              </h2>
              <div className="flex items-center gap-2 text-xs text-slate-500 font-medium">
                <Clock className="w-3.5 h-3.5" />
                <span>Elapsed: {status.elapsed_seconds.toFixed(1)}s</span>
                <span className="text-slate-300">•</span>
                <span className="text-cyan-accent font-bold">{status.progress_percentage}% Complete</span>
              </div>
            </div>
          </div>

          {isComplete && (
            <button
              onClick={onClose}
              className="px-4 py-1.5 rounded-xl bg-violet-primary hover:bg-violet-hover text-white text-xs font-bold shadow-violet-glow transition-all"
            >
              Open in 3D Studio
            </button>
          )}
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden border border-slate-200">
          <div
            className={`h-full transition-all duration-300 ${
              isFailed
                ? 'bg-red-500'
                : 'bg-gradient-to-r from-violet-primary via-indigo-600 to-cyan-accent'
            }`}
            style={{ width: `${Math.max(5, status.progress_percentage)}%` }}
          />
        </div>

        {/* Error Alert */}
        {isFailed && (
          <div className="p-3.5 rounded-2xl bg-red-50 border border-red-200 text-red-700 text-xs flex flex-col gap-2">
            <div className="flex items-center gap-2 font-bold">
              <AlertCircle className="w-4 h-4 text-red-600" />
              <span>Pipeline Error Occurred</span>
            </div>
            <p className="text-[11px] text-red-800 font-mono bg-red-100/60 p-2 rounded-lg">
              {status.error || 'An unexpected error occurred during mesh processing.'}
            </p>
            <div className="flex justify-end gap-2 pt-1">
              <button
                onClick={onRetry}
                className="px-3 py-1.5 rounded-xl bg-red-600 hover:bg-red-700 text-white text-xs font-bold flex items-center gap-1.5 transition-all"
              >
                <RotateCcw className="w-3 h-3" />
                <span>Retry</span>
              </button>
              <button
                onClick={onClose}
                className="px-3 py-1.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-bold transition-all"
              >
                Dismiss
              </button>
            </div>
          </div>
        )}

        {/* 8-Stage Tracker */}
        <div className="flex flex-col gap-2 max-h-64 overflow-y-auto pr-1">
          {STAGES_LIST.map((stage, index) => {
            const isStageComplete = status.stage_index > index || isComplete;
            const isCurrent = status.stage_index === index && !isComplete && !isFailed;

            return (
              <div
                key={stage.key}
                className={`flex items-start gap-3 p-2.5 rounded-xl border transition-all ${
                  isCurrent
                    ? 'border-violet-primary/50 bg-violet-dim/50 shadow-xs'
                    : isStageComplete
                    ? 'border-slate-200 bg-slate-50'
                    : 'border-transparent bg-transparent opacity-40'
                }`}
              >
                <div className="mt-0.5 shrink-0">
                  {isStageComplete ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  ) : isCurrent ? (
                    <Loader2 className="w-4 h-4 text-violet-primary animate-spin" />
                  ) : (
                    <div className="w-4 h-4 rounded-full border border-slate-300" />
                  )}
                </div>
                <div className="flex flex-col text-left">
                  <span className={`text-xs font-bold ${isCurrent ? 'text-violet-primary' : isStageComplete ? 'text-slate-800' : 'text-slate-400'}`}>
                    {stage.label}
                  </span>
                  <span className="text-[11px] text-slate-500 font-medium mt-0.5">
                    {stage.desc}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Honest Architecture Label */}
        <div className="p-3 rounded-2xl bg-slate-50 border border-slate-200 text-[11px] text-slate-600 flex items-center justify-between font-medium">
          <span>Engine: Watertight SDF + Marching Cubes</span>
          <span className="text-cyan-accent font-bold">Genuine 3D Geometry</span>
        </div>

      </div>
    </div>
  );
};

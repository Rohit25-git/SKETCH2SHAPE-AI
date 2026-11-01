import React, { useState } from 'react';
import { 
  Paintbrush, 
  UploadCloud, 
  PlusCircle, 
  Box, 
  History
} from 'lucide-react';
import { SketchCanvas } from '../components/studio/SketchCanvas';
import { ImageUpload } from '../components/studio/ImageUpload';
import { GenerationControls } from '../components/studio/GenerationControls';
import { GenerationModal } from '../components/studio/GenerationModal';
import { Viewer3D } from '../components/studio/Viewer3D';
import { api } from '../services/api';
import type { JobStatus, ModelMetadata, SystemHealth } from '../types';

interface WorkspacePageProps {
  onOpenCreations: () => void;
  health: SystemHealth | null;
}

export const WorkspacePage: React.FC<WorkspacePageProps> = ({
  onOpenCreations,
  health,
}) => {
  // Input mode
  const [activeInputTab, setActiveInputTab] = useState<'canvas' | 'upload'>('canvas');

  // Sketch data
  const [canvasBase64, setCanvasBase64] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState<string | null>(null);

  // Directives
  const [prompt, setPrompt] = useState('');
  const [engine, setEngine] = useState('local_manifold');
  const [resolution, setResolution] = useState('standard');
  const [smoothness, setSmoothness] = useState(3);
  const [thickness, setThickness] = useState(1.0);

  // Generation Job State
  const [isGenerating, setIsGenerating] = useState(false);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // 3D Result
  const [activeModel, setActiveModel] = useState<ModelMetadata | null>(null);
  const [activeModelUrl, setActiveModelUrl] = useState<string | null>(null);

  const hasInput = Boolean(
    (activeInputTab === 'canvas' && canvasBase64) ||
    (activeInputTab === 'upload' && uploadedFile)
  );

  const handleNewProject = () => {
    setCanvasBase64(null);
    setUploadedFile(null);
    setUploadPreviewUrl(null);
    setPrompt('');
    setActiveModel(null);
    setActiveModelUrl(null);
    setJobStatus(null);
    setIsModalOpen(false);
  };

  const handleStartGeneration = async () => {
    if (!hasInput) return;
    setIsGenerating(true);
    setIsModalOpen(true);
    setActiveModel(null);
    setActiveModelUrl(null);

    try {
      let imageId = '';

      // Step 1: Upload or Save image
      if (activeInputTab === 'canvas' && canvasBase64) {
        const uploadRes = await api.uploadBase64(canvasBase64);
        imageId = uploadRes.image_id;
      } else if (activeInputTab === 'upload' && uploadedFile) {
        const uploadRes = await api.uploadFile(uploadedFile);
        imageId = uploadRes.image_id;
      }

      // Step 2: Initiate 3D generation job
      const initialJob = await api.generate3D(
        imageId,
        prompt,
        engine,
        resolution,
        smoothness,
        thickness
      );
      setJobStatus(initialJob);

      // Step 3: Poll status until completion or failure
      const pollInterval = setInterval(async () => {
        try {
          const currentStatus = await api.getJobStatus(initialJob.job_id);
          setJobStatus(currentStatus);

          if (currentStatus.status === 'completed' && currentStatus.model_id) {
            clearInterval(pollInterval);
            setIsGenerating(false);

            // Fetch final model metadata
            const metadata = await api.getModel(currentStatus.model_id);
            setActiveModel(metadata);
            setActiveModelUrl(`/api/model/${metadata.model_id}/download?format=glb&v=${Date.now()}`);
          } else if (currentStatus.status === 'failed') {
            clearInterval(pollInterval);
            setIsGenerating(false);
          }
        } catch (pollErr) {
          clearInterval(pollInterval);
          setIsGenerating(false);
        }
      }, 700);

    } catch (err: any) {
      setIsGenerating(false);
      setJobStatus({
        job_id: 'err',
        status: 'failed',
        stage: 'input_validation',
        stage_index: 0,
        total_stages: 8,
        progress_percentage: 0,
        elapsed_seconds: 0,
        error: err.message || 'Generation failed to start',
      });
    }
  };

  return (
    <div className="flex flex-col lg:flex-row min-h-[calc(100vh-65px)] max-w-7xl mx-auto px-4 lg:px-8 py-6 gap-6">
      
      {/* Left Navigation Sidebar */}
      <aside className="w-full lg:w-60 flex lg:flex-col gap-2 shrink-0">
        
        <button
          onClick={handleNewProject}
          className="w-full py-2.5 px-3.5 rounded-2xl bg-white hover:bg-slate-50 border border-slate-200 text-slate-800 text-xs font-bold flex items-center justify-center lg:justify-start gap-2 transition-all shadow-xs group"
        >
          <PlusCircle className="w-4 h-4 text-violet-primary group-hover:rotate-90 transition-transform" />
          <span>New Project</span>
        </button>

        <div className="w-px lg:w-full h-auto lg:h-px bg-slate-200 my-1" />

        <div className="flex lg:flex-col gap-1 w-full">
          <button
            onClick={() => setActiveInputTab('canvas')}
            className={`w-full py-2 px-3 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
              activeInputTab === 'canvas'
                ? 'bg-violet-dim text-violet-primary border border-violet-primary/30'
                : 'text-slate-600 hover:text-slate-900 hover:bg-white'
            }`}
          >
            <Paintbrush className="w-3.5 h-3.5 text-violet-primary" />
            <span className="hidden sm:inline">Draw Sketch</span>
          </button>

          <button
            onClick={() => setActiveInputTab('upload')}
            className={`w-full py-2 px-3 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
              activeInputTab === 'upload'
                ? 'bg-violet-dim text-violet-primary border border-violet-primary/30'
                : 'text-slate-600 hover:text-slate-900 hover:bg-white'
            }`}
          >
            <UploadCloud className="w-3.5 h-3.5 text-cyan-accent" />
            <span className="hidden sm:inline">Upload Image</span>
          </button>

          <button
            onClick={onOpenCreations}
            className="w-full py-2 px-3 rounded-xl text-xs font-semibold text-slate-600 hover:text-slate-900 hover:bg-white flex items-center gap-2 transition-all"
          >
            <History className="w-3.5 h-3.5 text-amber-500" />
            <span className="hidden sm:inline">My Creations</span>
          </button>
        </div>

        {/* Engine status indicator box */}
        <div className="hidden lg:flex flex-col gap-2 p-3 mt-auto rounded-2xl bg-white border border-slate-200 text-[11px] text-slate-600 shadow-xs">
          <div className="flex items-center justify-between font-bold text-slate-900">
            <span className="flex items-center gap-1.5">
              <Box className="w-3.5 h-3.5 text-cyan-accent" />
              <span>Pipeline Status</span>
            </span>
            <span className="text-[10px] text-emerald-600 font-semibold flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              Online
            </span>
          </div>
          <p className="text-slate-500 text-[10px] leading-tight font-medium">
            Watertight Marching Cubes mesh reconstruction ready.
          </p>
        </div>

      </aside>

      {/* Main Workspace Layout (Central Canvas/Upload & Right Viewer/Controls) */}
      <main className="flex-1 grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        
        {/* Central Creation Area (Col 1-6) */}
        <div className="xl:col-span-5 flex flex-col gap-4">
          
          {/* Input Method Switcher */}
          <div className="flex items-center justify-between p-1.5 rounded-2xl bg-studio-surface border border-studio-border">
            <button
              onClick={() => setActiveInputTab('canvas')}
              className={`flex-1 py-1.5 rounded-xl text-xs font-medium transition-all flex items-center justify-center gap-2 ${
                activeInputTab === 'canvas'
                  ? 'bg-violet-primary text-white shadow-violet-glow'
                  : 'text-studio-muted hover:text-white'
              }`}
            >
              <Paintbrush className="w-3.5 h-3.5" />
              <span>Canvas Brush</span>
            </button>

            <button
              onClick={() => setActiveInputTab('upload')}
              className={`flex-1 py-1.5 rounded-xl text-xs font-medium transition-all flex items-center justify-center gap-2 ${
                activeInputTab === 'upload'
                  ? 'bg-violet-primary text-white shadow-violet-glow'
                  : 'text-studio-muted hover:text-white'
              }`}
            >
              <UploadCloud className="w-3.5 h-3.5" />
              <span>Upload Sketch</span>
            </button>
          </div>

          {/* Active Input Tool */}
          {activeInputTab === 'canvas' ? (
            <SketchCanvas onCanvasChange={(b64) => setCanvasBase64(b64)} />
          ) : (
            <ImageUpload
              previewUrl={uploadPreviewUrl}
              onImageSelected={(file) => {
                setUploadedFile(file);
                setUploadPreviewUrl(URL.createObjectURL(file));
              }}
              onClear={() => {
                setUploadedFile(null);
                setUploadPreviewUrl(null);
              }}
            />
          )}

          {/* Directives & Generation Controls */}
          <GenerationControls
            prompt={prompt}
            setPrompt={setPrompt}
            engine={engine}
            setEngine={setEngine}
            resolution={resolution}
            setResolution={setResolution}
            smoothness={smoothness}
            setSmoothness={setSmoothness}
            thickness={thickness}
            setThickness={setThickness}
            onGenerate={handleStartGeneration}
            isGenerating={isGenerating}
            hasInput={hasInput}
            health={health}
          />

        </div>

        {/* Right Output Area: Interactive 3D Studio (Col 7-12) */}
        <div className="xl:col-span-7 flex flex-col gap-4">
          <Viewer3D
            modelMetadata={activeModel}
            modelUrl={activeModelUrl}
            onReset={handleNewProject}
          />
        </div>

      </main>

      {/* Real-time 8-Stage Progress Modal */}
      <GenerationModal
        isOpen={isModalOpen}
        status={jobStatus}
        onRetry={handleStartGeneration}
        onClose={() => setIsModalOpen(false)}
      />

    </div>
  );
};

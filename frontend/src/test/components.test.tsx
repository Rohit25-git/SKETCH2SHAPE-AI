import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { Navbar } from '../components/common/Navbar';
import { SketchCanvas } from '../components/studio/SketchCanvas';
import { ImageUpload } from '../components/studio/ImageUpload';
import { GenerationControls } from '../components/studio/GenerationControls';
import { GenerationModal } from '../components/studio/GenerationModal';

describe('Frontend Component Tests', () => {

  it('renders Navbar with branding and navigation tabs', () => {
    const setCurrentTab = vi.fn();
    render(
      <Navbar
        currentTab="landing"
        setCurrentTab={setCurrentTab}
        health={{
          status: 'healthy',
          version: '1.0.0',
          hardware: {
            os: 'Windows',
            cpu_count: 8,
            ram_total_gb: 16,
            ram_available_gb: 8,
            gpu_available: false,
            cuda_available: false,
          },
          available_engines: ['local_manifold'],
          cloud_api_configured: { tripo: false, meshy: false },
          groq_vision_configured: true,
          groq_vision_model: 'qwen/qwen3.8-27b (Groq LPU Vision)',
        }}
      />
    );

    expect(screen.getByText('SKETCH2SHAPE')).toBeInTheDocument();
    expect(screen.getByText('Studio')).toBeInTheDocument();
    expect(screen.getByText('Creations')).toBeInTheDocument();

    const studioBtn = screen.getByText('Studio');
    fireEvent.click(studioBtn);
    expect(setCurrentTab).toHaveBeenCalledWith('workspace');
  });

  it('renders SketchCanvas with drawing tools and size slider', () => {
    const onCanvasChange = vi.fn();
    render(<SketchCanvas onCanvasChange={onCanvasChange} />);

    expect(screen.getByText('Brush')).toBeInTheDocument();
    expect(screen.getByText('Eraser')).toBeInTheDocument();
    expect(screen.getByText('Size:')).toBeInTheDocument();

    // Toggle eraser
    const eraserBtn = screen.getByText('Eraser');
    fireEvent.click(eraserBtn);
    expect(eraserBtn.closest('button')).toHaveClass('bg-violet-primary');
  });

  it('validates file uploads and rejects bad formats', () => {
    const onImageSelected = vi.fn();
    const onClear = vi.fn();
    render(
      <ImageUpload
        previewUrl={null}
        onImageSelected={onImageSelected}
        onClear={onClear}
      />
    );

    expect(screen.getByText('Drag & drop your sketch here')).toBeInTheDocument();

    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    const badFile = new File(['bad'], 'script.exe', { type: 'application/x-msdownload' });
    
    fireEvent.change(input, { target: { files: [badFile] } });
    expect(screen.getByText(/Invalid file format/i)).toBeInTheDocument();
    expect(onImageSelected).not.toHaveBeenCalled();
  });

  it('renders GenerationControls and triggers generation', () => {
    const onGenerate = vi.fn();
    render(
      <GenerationControls
        prompt="A wooden armchair"
        setPrompt={vi.fn()}
        engine="local_manifold"
        setEngine={vi.fn()}
        resolution="standard"
        setResolution={vi.fn()}
        smoothness={3}
        setSmoothness={vi.fn()}
        thickness={1.0}
        setThickness={vi.fn()}
        onGenerate={onGenerate}
        isGenerating={false}
        hasInput={true}
        health={null}
      />
    );

    const generateBtn = screen.getByRole('button', { name: /generate 3d model/i });
    expect(generateBtn).not.toBeDisabled();
    fireEvent.click(generateBtn);
    expect(onGenerate).toHaveBeenCalled();
  });

  it('renders GenerationModal with 8 honest stages and elapsed time', () => {
    render(
      <GenerationModal
        isOpen={true}
        status={{
          job_id: 'job_test_123',
          status: 'processing',
          stage: '3d_geometry_generation',
          stage_index: 4,
          total_stages: 8,
          progress_percentage: 70,
          elapsed_seconds: 3.4,
        }}
        onRetry={vi.fn()}
        onClose={vi.fn()}
      />
    );

    expect(screen.getByText('Synthesizing 3D Asset')).toBeInTheDocument();
    expect(screen.getByText(/Elapsed: 3.4s/i)).toBeInTheDocument();
    expect(screen.getByText('5. Geometry Generation')).toBeInTheDocument();
  });

});

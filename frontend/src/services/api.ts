import type {
  UploadResult,
  PreprocessResult,
  AIAnalysis,
  JobStatus,
  ModelMetadata,
  HistoryItem,
  SystemHealth
} from '../types';

const API_BASE = '/api';

export const api = {
  // Health
  async getHealth(): Promise<SystemHealth> {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Failed to fetch system health');
    return res.json();
  },

  // Upload file (Multipart)
  async uploadFile(file: File): Promise<UploadResult> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/sketch/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  // Upload Base64 Sketch Canvas
  async uploadBase64(base64Data: string, filename = 'sketch.png'): Promise<UploadResult> {
    const res = await fetch(`${API_BASE}/sketch/upload/base64`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: base64Data, filename }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Base64 upload failed' }));
      throw new Error(err.detail || 'Base64 upload failed');
    }
    return res.json();
  },

  // Preprocess Image
  async preprocess(imageId: string, threshold = 220, removeBg = true, invert = false): Promise<PreprocessResult> {
    const res = await fetch(`${API_BASE}/image/preprocess`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_id: imageId,
        threshold,
        remove_background: removeBg,
        invert,
        edge_detection: true,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Preprocessing failed' }));
      throw new Error(err.detail || 'Preprocessing failed');
    }
    return res.json();
  },

  // Groq Multimodal Vision AI Analysis
  async analyzeSketch(imageId: string, prompt?: string): Promise<AIAnalysis> {
    const res = await fetch(`${API_BASE}/sketch/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_id: imageId, prompt }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Vision analysis failed' }));
      throw new Error(err.detail || 'Vision analysis failed');
    }
    return res.json();
  },

  // Trigger 3D Generation Job
  async generate3D(
    imageId: string,
    prompt?: string,
    engine = 'local_manifold',
    resolution = 'standard',
    smoothness = 3,
    thickness = 1.0
  ): Promise<JobStatus> {
    const res = await fetch(`${API_BASE}/generate/3d`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_id: imageId,
        prompt: prompt || undefined,
        engine,
        resolution,
        smoothness,
        thickness,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Generation initiation failed' }));
      throw new Error(err.detail || 'Generation initiation failed');
    }
    return res.json();
  },

  // Poll Job Status
  async getJobStatus(jobId: string): Promise<JobStatus> {
    const res = await fetch(`${API_BASE}/generate/status/${jobId}`);
    if (!res.ok) throw new Error('Job status check failed');
    return res.json();
  },

  // Model Details
  async getModel(modelId: string): Promise<ModelMetadata> {
    const res = await fetch(`${API_BASE}/model/${modelId}`);
    if (!res.ok) throw new Error('Model metadata not found');
    return res.json();
  },

  // Download URLs
  getDownloadUrl(modelId: string, format: 'glb' | 'obj'): string {
    return `${API_BASE}/model/${modelId}/download?format=${format}`;
  },

  // History
  async getHistory(): Promise<HistoryItem[]> {
    const res = await fetch(`${API_BASE}/history`);
    if (!res.ok) throw new Error('Failed to fetch history');
    return res.json();
  },

  // Delete History
  async deleteHistory(modelId: string): Promise<boolean> {
    const res = await fetch(`${API_BASE}/history/${modelId}`, {
      method: 'DELETE',
    });
    return res.ok;
  },
};

export interface GeometryMetrics {
  vertex_count: number;
  face_count: number;
  is_watertight: boolean;
  dimensions: [number, number, number];
  volume: number;
  surface_area: number;
}

export interface AIAnalysis {
  available: boolean;
  object_type: string;
  description: string;
  suggested_thickness: number;
  suggested_smoothness: number;
  visual_features: string[];
  object_category?: string;
  object_name?: string;
  main_body?: {
    shape?: string;
    proportions?: string;
    top?: string;
    interior?: string;
    base?: string;
  };
  components?: Array<{
    name?: string;
    shape?: string;
    position?: string;
    attachment?: string;
    preserve_opening?: boolean;
  }>;
  symmetry?: string;
  view?: string;
  material?: string;
  primary_color?: string;
  roughness?: number;
  metalness?: number;
  confidence?: number;
  model_used?: string;
  error?: string;
}

export interface MaterialProps {
  name?: string;
  base_color?: string;
  roughness?: number;
  metallic?: number;
  metalness?: number;
  is_transparent?: boolean;
  paradigm?: string;
}

export interface QualityValidation {
  geometry_validity: boolean;
  object_recognition_confidence: number;
  component_completeness: number;
  material_completeness: number;
  visual_reconstruction_quality: string;
  quality_score: number;
  validated_components: Array<{
    component: string;
    status: string;
    submesh_count?: number;
    material?: string;
  }>;
}

export interface IdentityTracing {
  image_id: string;
  job_id: string;
  original_filename: string;
  image_hash_sha256: string;
  original_dimensions: [number, number];
  preprocessed_path: string;
  preprocessed_hash: string;
  vision_model: string;
  detected_object: string;
  detected_category: string;
  detection_confidence: number;
  final_prompt: string;
  provider_task_id: string;
  downloaded_asset_path: string;
  final_glb_hash: string;
  viewer_asset_url: string;
  validation_status: string;
}

export interface ModelMetadata {
  model_id: string;
  job_id?: string;
  prompt?: string;
  engine: string;
  created_at: string;
  processing_time_seconds: number;
  geometry: GeometryMetrics;
  ai_analysis?: AIAnalysis;
  material_props?: MaterialProps;
  quality_validation?: QualityValidation;
  identity_tracing?: IdentityTracing;
  preview_image_url: string;
  glb_url: string;
  obj_url: string;
}

export interface HistoryItem {
  model_id: string;
  prompt?: string;
  engine: string;
  created_at: string;
  vertex_count: number;
  face_count: number;
  ai_object_type?: string;
  preview_image_url: string;
  glb_url: string;
  obj_url: string;
}

export interface JobStatus {
  job_id: string;
  model_id?: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  stage: string;
  stage_index: number;
  total_stages: number;
  progress_percentage: number;
  elapsed_seconds: number;
  error?: string;
  metadata?: ModelMetadata;
}

export interface HardwareInfo {
  os: string;
  cpu_count: number;
  ram_total_gb: number;
  ram_available_gb: number;
  gpu_available: boolean;
  gpu_name?: string;
  cuda_available: boolean;
  torch_version?: string;
}

export interface SystemHealth {
  status: string;
  version: string;
  hardware: HardwareInfo;
  available_engines: string[];
  cloud_api_configured: {
    tripo: boolean;
    meshy: boolean;
  };
  groq_vision_configured: boolean;
  groq_vision_model: string;
}

export interface UploadResult {
  image_id: string;
  filename: string;
  url: string;
  width: number;
  height: number;
  size_bytes: number;
}

export interface PreprocessResult {
  processed_id: string;
  url: string;
  preview_url: string;
  width: number;
  height: number;
  contours_detected: number;
  coverage_ratio: number;
}

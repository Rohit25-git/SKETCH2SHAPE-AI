import React, { useEffect, useState } from 'react';
import { 
  Box, 
  Download, 
  Trash2, 
  Eye, 
  Calendar, 
  RefreshCw,
  ArrowLeft
} from 'lucide-react';
import { api } from '../services/api';
import type { HistoryItem } from '../types';

interface CreationsPageProps {
  onOpenModel: (modelId: string) => void;
  onBackToStudio: () => void;
}

export const CreationsPage: React.FC<CreationsPageProps> = ({
  onOpenModel,
  onBackToStudio,
}) => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const items = await api.getHistory();
      setHistory(items);
    } catch (err) {
      console.error('Failed to load history', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleDelete = async (modelId: string) => {
    if (!confirm('Are you sure you want to delete this 3D model from storage?')) return;
    setDeletingId(modelId);
    try {
      await api.deleteHistory(modelId);
      setHistory(prev => prev.filter(item => item.model_id !== modelId));
    } catch (err) {
      alert('Failed to delete model');
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 lg:px-8 py-8 flex flex-col gap-8">
      
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-slate-200">
        <div className="flex items-center gap-3">
          <button
            onClick={onBackToStudio}
            className="p-2 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 hover:text-slate-900 transition-all shadow-xs"
            title="Back to Studio"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <div>
            <h1 className="text-2xl font-extrabold text-slate-900 flex items-center gap-2">
              <span>My Creations</span>
              <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-violet-dim text-violet-primary border border-violet-primary/30">
                {history.length} Assets
              </span>
            </h1>
            <p className="text-xs text-slate-500 font-medium mt-0.5">
              Persistent archive of all generated watertight 3D models.
            </p>
          </div>
        </div>

        <button
          onClick={fetchHistory}
          className="px-3.5 py-2 rounded-xl bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 hover:text-slate-900 text-xs font-bold flex items-center gap-2 transition-all shadow-xs"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Gallery Content */}
      {loading ? (
        <div className="flex flex-col items-center justify-center py-20 text-center gap-3">
          <RefreshCw className="w-8 h-8 text-violet-primary animate-spin" />
          <p className="text-sm text-slate-600 font-medium">Loading asset gallery...</p>
        </div>
      ) : history.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 text-center max-w-md mx-auto p-8 rounded-3xl bg-white border border-slate-200 shadow-sm">
          <div className="w-16 h-16 rounded-2xl bg-slate-50 border border-slate-200 flex items-center justify-center text-slate-400 mb-4">
            <Box className="w-8 h-8" />
          </div>
          <h3 className="text-base font-bold text-slate-900 mb-1">No 3D Models Generated Yet</h3>
          <p className="text-xs text-slate-500 mb-6 font-medium">
            Draw your first sketch or upload an illustration in the studio to generate interactive 3D assets.
          </p>
          <button
            onClick={onBackToStudio}
            className="px-5 py-2.5 rounded-xl bg-violet-primary hover:bg-violet-hover text-white text-xs font-bold shadow-violet-glow transition-all cursor-pointer"
          >
            Open Creative Studio
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {history.map((item) => (
            <div
              key={item.model_id}
              className="p-5 rounded-3xl bg-white border border-slate-200 flex flex-col gap-4 shadow-studio-card hover:border-violet-primary/50 transition-all group"
            >
              {/* Preview Thumbnail */}
              <div className="relative aspect-video rounded-2xl bg-slate-50 overflow-hidden border border-slate-200 flex items-center justify-center">
                {item.preview_image_url ? (
                  <img
                    src={item.preview_image_url}
                    alt={item.model_id}
                    className="w-full h-full object-contain p-2"
                  />
                ) : (
                  <Box className="w-12 h-12 text-slate-300" />
                )}

                <div className="absolute top-2.5 right-2.5 px-2 py-0.5 rounded-full bg-white/95 backdrop-blur text-[10px] text-cyan-accent font-bold border border-slate-200 shadow-xs">
                  {item.engine === 'local_manifold' ? 'Watertight SDF' : 'Cloud AI'}
                </div>
              </div>

              {/* Information */}
              <div className="flex flex-col gap-1.5 text-left">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-mono text-slate-900 font-bold truncate max-w-[180px]">
                    {item.model_id}
                  </span>
                  <span className="text-[10px] text-slate-500 font-medium flex items-center gap-1">
                    <Calendar className="w-3 h-3" />
                    <span>{new Date(item.created_at).toLocaleDateString()}</span>
                  </span>
                </div>

                {item.prompt ? (
                  <p className="text-xs text-slate-600 italic line-clamp-1 font-medium">
                    &ldquo;{item.prompt}&rdquo;
                  </p>
                ) : (
                  <p className="text-xs text-slate-400 font-medium">Sketch Reconstruction</p>
                )}

                {/* Geometry HUD */}
                <div className="grid grid-cols-2 gap-2 mt-2 pt-2 border-t border-slate-100 text-[11px] text-slate-600">
                  <div>
                    <span className="text-slate-400 block text-[10px] font-medium">Vertices</span>
                    <span className="text-slate-900 font-mono font-bold">
                      {item.vertex_count.toLocaleString()}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400 block text-[10px] font-medium">Faces</span>
                    <span className="text-slate-900 font-mono font-bold">
                      {item.face_count.toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-2 pt-2 border-t border-slate-100">
                <button
                  onClick={() => onOpenModel(item.model_id)}
                  className="flex-1 py-2 rounded-xl bg-violet-primary hover:bg-violet-hover text-white text-xs font-bold shadow-violet-glow flex items-center justify-center gap-1.5 transition-all cursor-pointer"
                >
                  <Eye className="w-3.5 h-3.5" />
                  <span>Inspect in 3D</span>
                </button>

                <a
                  href={api.getDownloadUrl(item.model_id, 'glb')}
                  download={`sketch2shape_${item.model_id}.glb`}
                  className="p-2 rounded-xl bg-slate-50 hover:bg-slate-100 text-slate-700 hover:text-slate-900 border border-slate-200 transition-all shadow-xs"
                  title="Download GLB"
                >
                  <Download className="w-3.5 h-3.5" />
                </a>

                <button
                  onClick={() => handleDelete(item.model_id)}
                  disabled={deletingId === item.model_id}
                  className="p-2 rounded-xl bg-slate-50 hover:bg-red-50 text-slate-400 hover:text-red-500 border border-slate-200 transition-all shadow-xs"
                  title="Delete Entry"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>

            </div>
          ))}
        </div>
      )}

    </div>
  );
};

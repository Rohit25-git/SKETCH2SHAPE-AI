import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, X, RefreshCw, AlertCircle } from 'lucide-react';

interface ImageUploadProps {
  onImageSelected: (file: File) => void;
  onClear: () => void;
  previewUrl: string | null;
}

export const ImageUpload: React.FC<ImageUploadProps> = ({
  onImageSelected,
  onClear,
  previewUrl,
}) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const validateAndHandle = (file: File) => {
    setErrorMessage(null);
    const validExtensions = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];

    if (!validExtensions.includes(file.type)) {
      setErrorMessage('Invalid file format. Please upload PNG, JPG, or WEBP.');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setErrorMessage('File size exceeds the 10MB limit.');
      return;
    }

    onImageSelected(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndHandle(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="w-full max-w-xl mx-auto flex flex-col gap-3">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/jpg,image/webp"
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            validateAndHandle(e.target.files[0]);
          }
        }}
      />

      {errorMessage && (
        <div className="flex items-center gap-2 p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      {!previewUrl ? (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative aspect-square w-full rounded-3xl border-2 border-dashed flex flex-col items-center justify-center p-8 text-center cursor-pointer transition-all ${
            isDragging
              ? 'border-violet-primary bg-violet-dim/60 scale-[0.99]'
              : 'border-slate-300 hover:border-violet-primary/60 bg-white hover:bg-slate-50/80 shadow-studio-card'
          }`}
        >
          <div className="w-16 h-16 rounded-2xl bg-slate-100 border border-slate-200 flex items-center justify-center text-violet-primary mb-4 shadow-xs">
            <UploadCloud className="w-8 h-8 animate-pulse" />
          </div>

          <h3 className="text-base font-bold text-slate-900 mb-1">
            Drag & drop your sketch here
          </h3>
          <p className="text-xs text-slate-500 mb-4 max-w-xs font-medium">
            Supports high-resolution PNG, JPG, or WEBP sketches up to 10MB.
          </p>

          <button
            type="button"
            className="px-4 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-800 text-xs font-bold border border-slate-200 transition-all flex items-center gap-2"
          >
            <ImageIcon className="w-3.5 h-3.5 text-cyan-accent" />
            <span>Browse Files</span>
          </button>
        </div>
      ) : (
        <div className="relative aspect-square w-full rounded-3xl border-2 border-slate-200 bg-white overflow-hidden shadow-studio-card flex items-center justify-center group">
          <img
            src={previewUrl}
            alt="Sketch Preview"
            className="w-full h-full object-contain p-6"
          />

          {/* Overlay Controls */}
          <div className="absolute top-4 right-4 flex items-center gap-2">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="p-2.5 rounded-xl bg-white/90 hover:bg-slate-100 text-slate-700 backdrop-blur border border-slate-200 shadow-xs transition-all"
              title="Replace Image"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              onClick={onClear}
              className="p-2.5 rounded-xl bg-white/90 hover:bg-red-50 text-slate-500 hover:text-red-500 backdrop-blur border border-slate-200 shadow-xs transition-all"
              title="Remove Image"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="absolute bottom-4 left-4 right-4 px-3 py-2 rounded-xl bg-white/95 backdrop-blur border border-slate-200 text-center shadow-xs">
            <span className="text-xs text-slate-600 font-medium">
              Sketch loaded & validated. Ready for 3D reconstruction.
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

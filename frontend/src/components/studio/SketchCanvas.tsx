import React, { useRef, useState, useEffect } from 'react';
import { 
  Paintbrush, 
  Eraser, 
  Undo2, 
  Redo2, 
  Trash2, 
  Download, 
  Grid3X3, 
  Sun, 
  Moon,
  Sparkles
} from 'lucide-react';

interface SketchCanvasProps {
  onCanvasChange: (base64: string) => void;
}

const PRESET_COLORS = [
  '#0F172A', // Pitch Black / Slate
  '#475569', // Graphite Grey
  '#6366F1', // Indigo
  '#0284C7', // Sky Blue
  '#0D9488', // Teal
  '#DC2626', // Crimson Red
  '#D97706', // Amber
  '#7C3AED', // Royal Violet
];

export const SketchCanvas: React.FC<SketchCanvasProps> = ({ onCanvasChange }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [tool, setTool] = useState<'brush' | 'eraser'>('brush');
  const [color, setColor] = useState('#0F172A');
  const [brushSize, setBrushSize] = useState(6);
  const [canvasBg, setCanvasBg] = useState<'dark' | 'grid' | 'light'>('light');

  // History stack for Undo / Redo
  const [history, setHistory] = useState<ImageData[]>([]);
  const [historyStep, setHistoryStep] = useState<number>(-1);

  // Initialize Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Handle high-DPI
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = 600 * dpr;
    canvas.height = 600 * dpr;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.scale(dpr, dpr);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Save initial blank state
    const initialState = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setHistory([initialState]);
    setHistoryStep(0);
  }, []);

  const saveToHistory = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const currentState = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const newHistory = history.slice(0, historyStep + 1);
    newHistory.push(currentState);
    
    // Limit stack size to 25
    if (newHistory.length > 25) newHistory.shift();

    setHistory(newHistory);
    setHistoryStep(newHistory.length - 1);

    // Notify parent
    exportBase64();
  };

  const undo = () => {
    if (historyStep <= 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const nextStep = historyStep - 1;
    ctx.putImageData(history[nextStep], 0, 0);
    setHistoryStep(nextStep);
    exportBase64();
  };

  const redo = () => {
    if (historyStep >= history.length - 1) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const nextStep = historyStep + 1;
    ctx.putImageData(history[nextStep], 0, 0);
    setHistoryStep(nextStep);
    exportBase64();
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    saveToHistory();
  };

  const exportBase64 = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    // Export with transparent background
    const base64 = canvas.toDataURL('image/png');
    onCanvasChange(base64);
  };

  const downloadPNG = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `sketch_${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  // Coordinates helper
  const getCoordinates = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();

    let clientX = 0;
    let clientY = 0;

    if ('touches' in e && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else if ('clientX' in e) {
      clientX = (e as React.MouseEvent).clientX;
      clientY = (e as React.MouseEvent).clientY;
    }

    return {
      x: (clientX - rect.left) * (600 / rect.width),
      y: (clientY - rect.top) * (600 / rect.height),
    };
  };

  const startDrawing = (e: React.MouseEvent | React.TouchEvent) => {
    setIsDrawing(true);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { x, y } = getCoordinates(e);

    ctx.beginPath();
    ctx.moveTo(x, y);

    if (tool === 'eraser') {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.lineWidth = brushSize * 2.5;
    } else {
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = color;
      ctx.lineWidth = brushSize;
    }
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { x, y } = getCoordinates(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.closePath();
    saveToHistory();
  };

  // Background styling
  const getBgStyle = () => {
    switch (canvasBg) {
      case 'dark':
        return 'bg-[#0F172A]';
      case 'grid':
        return 'bg-white bg-[radial-gradient(#CBD5E1_1px,transparent_1px)] [background-size:16px_16px]';
      case 'light':
      default:
        return 'bg-white';
    }
  };

  return (
    <div className="flex flex-col gap-3 w-full max-w-xl mx-auto">
      {/* Canvas Toolbar Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 p-2.5 rounded-2xl bg-studio-surface border border-studio-border">
        
        {/* Tool Selectors */}
        <div className="flex items-center gap-1 bg-studio-bg p-1 rounded-xl border border-studio-border/60">
          <button
            onClick={() => setTool('brush')}
            className={`p-2 rounded-lg text-xs flex items-center gap-1.5 transition-all ${
              tool === 'brush'
                ? 'bg-violet-primary text-white shadow-violet-glow'
                : 'text-studio-muted hover:text-white'
            }`}
            title="Brush"
          >
            <Paintbrush className="w-4 h-4" />
            <span className="hidden sm:inline font-medium">Brush</span>
          </button>

          <button
            onClick={() => setTool('eraser')}
            className={`p-2 rounded-lg text-xs flex items-center gap-1.5 transition-all ${
              tool === 'eraser'
                ? 'bg-violet-primary text-white shadow-violet-glow'
                : 'text-studio-muted hover:text-white'
            }`}
            title="Eraser"
          >
            <Eraser className="w-4 h-4" />
            <span className="hidden sm:inline font-medium">Eraser</span>
          </button>
        </div>

        {/* Size Slider */}
        <div className="flex items-center gap-2 px-3 py-1 bg-studio-bg rounded-xl border border-studio-border/60">
          <span className="text-[11px] font-medium text-studio-muted">Size:</span>
          <input
            type="range"
            min="1"
            max="40"
            value={brushSize}
            onChange={(e) => setBrushSize(Number(e.target.value))}
            className="w-20 accent-violet-primary cursor-pointer"
          />
          <span className="text-[11px] text-white font-mono w-4">{brushSize}</span>
        </div>

        {/* Background Mode Toggles */}
        <div className="flex items-center gap-1 bg-studio-bg p-1 rounded-xl border border-studio-border/60">
          <button
            onClick={() => setCanvasBg('dark')}
            className={`p-1.5 rounded-lg text-xs transition-all ${
              canvasBg === 'dark' ? 'bg-studio-panel text-white' : 'text-studio-muted hover:text-white'
            }`}
            title="Dark Background"
          >
            <Moon className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setCanvasBg('grid')}
            className={`p-1.5 rounded-lg text-xs transition-all ${
              canvasBg === 'grid' ? 'bg-studio-panel text-cyan-accent' : 'text-studio-muted hover:text-white'
            }`}
            title="Blueprint Grid"
          >
            <Grid3X3 className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setCanvasBg('light')}
            className={`p-1.5 rounded-lg text-xs transition-all ${
              canvasBg === 'light' ? 'bg-studio-panel text-amber-400' : 'text-studio-muted hover:text-white'
            }`}
            title="Light Background"
          >
            <Sun className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Actions (Undo, Redo, Clear, Download) */}
        <div className="flex items-center gap-1">
          <button
            onClick={undo}
            disabled={historyStep <= 0}
            className="p-2 rounded-xl bg-studio-bg hover:bg-studio-hover disabled:opacity-40 text-studio-muted hover:text-white border border-studio-border/60 transition-all"
            title="Undo"
          >
            <Undo2 className="w-4 h-4" />
          </button>

          <button
            onClick={redo}
            disabled={historyStep >= history.length - 1}
            className="p-2 rounded-xl bg-studio-bg hover:bg-studio-hover disabled:opacity-40 text-studio-muted hover:text-white border border-studio-border/60 transition-all"
            title="Redo"
          >
            <Redo2 className="w-4 h-4" />
          </button>

          <button
            onClick={clearCanvas}
            className="p-2 rounded-xl bg-studio-bg hover:bg-red-500/20 text-studio-muted hover:text-red-400 border border-studio-border/60 transition-all"
            title="Clear Canvas"
          >
            <Trash2 className="w-4 h-4" />
          </button>

          <button
            onClick={downloadPNG}
            className="p-2 rounded-xl bg-studio-bg hover:bg-cyan-accent/20 text-studio-muted hover:text-cyan-accent border border-studio-border/60 transition-all"
            title="Export PNG"
          >
            <Download className="w-4 h-4" />
          </button>
        </div>

      </div>

      {/* Preset Color Swatches */}
      {tool === 'brush' && (
        <div className="flex items-center gap-2 px-2 overflow-x-auto py-1">
          <span className="text-[11px] text-studio-muted">Color:</span>
          {PRESET_COLORS.map((c) => (
            <button
              key={c}
              onClick={() => setColor(c)}
              style={{ backgroundColor: c }}
              className={`w-6 h-6 rounded-full border transition-all ${
                color === c ? 'scale-125 border-violet-primary ring-2 ring-violet-primary/30' : 'border-slate-300 hover:scale-110'
              }`}
            />
          ))}
          <input
            type="color"
            value={color}
            onChange={(e) => setColor(e.target.value)}
            className="w-7 h-7 rounded-full bg-transparent border-0 cursor-pointer"
            title="Custom Color"
          />
        </div>
      )}

      {/* Main Drawing Canvas Area */}
      <div className={`relative w-full aspect-square rounded-3xl border-2 border-studio-border shadow-studio-card overflow-hidden transition-colors ${getBgStyle()}`}>
        <canvas
          ref={canvasRef}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
          className="w-full h-full cursor-crosshair touch-none"
        />

        {/* Empty State Prompt overlay when no strokes drawn */}
        {historyStep <= 0 && (
          <div className="absolute inset-0 pointer-events-none flex flex-col items-center justify-center text-center p-6 bg-slate-900/5 backdrop-blur-[1px]">
            <Sparkles className="w-8 h-8 text-violet-primary mb-2 animate-bounce" />
            <p className="text-sm font-bold text-slate-800">
              Draw your sketch here
            </p>
            <p className="text-xs text-slate-500 font-medium mt-1 max-w-xs">
              Draw any chair, cup, vase, vehicle, or character. Clean outlines reconstruct into watertight 3D models!
            </p>
          </div>
        )}
      </div>

    </div>
  );
};

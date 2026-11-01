import React from 'react';
import { Box, Sparkles, Layers, History, Cpu, ArrowRight } from 'lucide-react';
import type { SystemHealth } from '../../types';

interface NavbarProps {
  currentTab: 'landing' | 'workspace' | 'creations' | 'system';
  setCurrentTab: (tab: 'landing' | 'workspace' | 'creations' | 'system') => void;
  health: SystemHealth | null;
}

export const Navbar: React.FC<NavbarProps> = ({ currentTab, setCurrentTab, health }) => {
  return (
    <header className="sticky top-0 z-50 backdrop-blur-xl bg-studio-bg/90 border-b border-studio-border px-4 lg:px-8 py-3.5 transition-all">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        
        {/* Brand Logo */}
        <div 
          onClick={() => setCurrentTab('landing')}
          className="flex items-center gap-3 cursor-pointer group"
        >
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-violet-primary via-indigo-500 to-cyan-accent flex items-center justify-center shadow-violet-glow group-hover:scale-105 transition-transform duration-300">
            <Box className="w-5 h-5 text-white animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-lg tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-slate-900 via-indigo-900 to-violet-primary">
                SKETCH2SHAPE
              </span>
              <span className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-violet-dim border border-violet-primary/30 text-violet-primary">
                AI
              </span>
            </div>
            <p className="text-[11px] text-studio-muted hidden sm:block tracking-wide font-medium">
              Dimensional Reality Studio
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 sm:gap-2 bg-studio-surface border border-studio-border p-1 rounded-xl shadow-xs">
          <button
            onClick={() => setCurrentTab('landing')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
              currentTab === 'landing'
                ? 'bg-violet-primary text-white shadow-violet-glow'
                : 'text-studio-muted hover:text-studio-text hover:bg-studio-hover'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span className="hidden md:inline">Overview</span>
          </button>

          <button
            onClick={() => setCurrentTab('workspace')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
              currentTab === 'workspace'
                ? 'bg-violet-primary text-white shadow-violet-glow'
                : 'text-studio-muted hover:text-studio-text hover:bg-studio-hover'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Studio</span>
          </button>

          <button
            onClick={() => setCurrentTab('creations')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
              currentTab === 'creations'
                ? 'bg-violet-primary text-white shadow-violet-glow'
                : 'text-studio-muted hover:text-studio-text hover:bg-studio-hover'
            }`}
          >
            <History className="w-3.5 h-3.5" />
            <span>Creations</span>
          </button>

          <button
            onClick={() => setCurrentTab('system')}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 ${
              currentTab === 'system'
                ? 'bg-violet-primary text-white shadow-violet-glow'
                : 'text-studio-muted hover:text-studio-text hover:bg-studio-hover'
            }`}
          >
            <Cpu className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Engine</span>
          </button>
        </nav>

        {/* Right CTA & Health Badge */}
        <div className="flex items-center gap-3">
          {health && (
            <div className="hidden lg:flex items-center gap-2 px-3 py-1 rounded-full bg-studio-surface border border-studio-border text-[11px] text-studio-muted shadow-xs">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
              <span className="font-medium text-studio-text">{health.hardware.cpu_count} Cores</span>
              <span className="text-studio-dim">|</span>
              <span className="text-cyan-accent font-semibold">Manifold Ready</span>
            </div>
          )}

          {currentTab !== 'workspace' && (
            <button
              onClick={() => setCurrentTab('workspace')}
              className="px-4 py-2 rounded-xl bg-gradient-to-r from-violet-primary to-violet-hover hover:from-violet-hover hover:to-violet-glow text-white text-xs font-semibold shadow-violet-glow transition-all flex items-center gap-2 group"
            >
              <span>Launch Studio</span>
              <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
            </button>
          )}
        </div>

      </div>
    </header>
  );
};

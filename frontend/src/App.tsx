import React, { useState, useEffect } from 'react';
import { Navbar } from './components/common/Navbar';
import { LandingPage } from './pages/LandingPage';
import { WorkspacePage } from './pages/WorkspacePage';
import { CreationsPage } from './pages/CreationsPage';
import { SystemPage } from './pages/SystemPage';
import { api } from './services/api';
import type { SystemHealth } from './types';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState<'landing' | 'workspace' | 'creations' | 'system'>('landing');
  const [health, setHealth] = useState<SystemHealth | null>(null);

  useEffect(() => {
    // Fetch initial health metrics
    api.getHealth()
      .then(setHealth)
      .catch((err) => console.log('Backend offline or initializing...', err));
  }, []);

  return (
    <div className="min-h-screen bg-studio-bg text-studio-text flex flex-col font-sans selection:bg-violet-primary selection:text-white">
      {/* Top Studio Navbar */}
      <Navbar
        currentTab={currentTab}
        setCurrentTab={setCurrentTab}
        health={health}
      />

      {/* Dynamic Content Views */}
      <main className="flex-1 w-full">
        {currentTab === 'landing' && (
          <LandingPage
            onStartCreating={() => setCurrentTab('workspace')}
            onExploreFeatures={() => {
              const el = document.getElementById('features');
              if (el) el.scrollIntoView({ behavior: 'smooth' });
            }}
          />
        )}

        {currentTab === 'workspace' && (
          <WorkspacePage
            onOpenCreations={() => setCurrentTab('creations')}
            health={health}
          />
        )}

        {currentTab === 'creations' && (
          <CreationsPage
            onOpenModel={(_id) => {
              setCurrentTab('workspace');
            }}
            onBackToStudio={() => setCurrentTab('workspace')}
          />
        )}

        {currentTab === 'system' && (
          <SystemPage health={health} />
        )}
      </main>
    </div>
  );
};

export default App;

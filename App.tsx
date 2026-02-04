import React, { useState } from 'react';
import { SetupScreen } from './components/SetupScreen';
import { WorldViewport } from './components/WorldViewport';
import { SessionConfig } from './types';

const App: React.FC = () => {
  const [sessionConfig, setSessionConfig] = useState<SessionConfig | null>(null);

  const handleStartSession = (config: SessionConfig) => {
    setSessionConfig(config);
  };

  const handleExitSession = () => {
    setSessionConfig(null);
  };

  return (
    <div className="antialiased text-slate-200 selection:bg-cyan-500/30">
      {!sessionConfig ? (
        <SetupScreen onStart={handleStartSession} />
      ) : (
        <WorldViewport config={sessionConfig} onExit={handleExitSession} />
      )}
    </div>
  );
};

export default App;
import React, { useState } from 'react';
import { SetupScreen } from './components/SetupScreen';
import { WorldViewport } from './components/WorldViewport';
import { ActiveSession, CreateSessionResponse, SessionConfig } from './types';
import { API_BASE_URL } from './constants';

const App: React.FC = () => {
  const [activeSession, setActiveSession] = useState<ActiveSession | null>(null);

  const handleStartSession = async (config: SessionConfig) => {
    const formData = new FormData();
    formData.append('prompt', config.prompt);
    formData.append('resolution', config.resolution);
    formData.append('quality', config.quality);

    if (config.initImage) {
      formData.append('initImage', config.initImage);
    }

    const response = await fetch(`${API_BASE_URL}/api/session`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let detail = `Failed to create session (${response.status})`;
      try {
        const body = await response.json();
        if (typeof body?.detail === 'string') {
          detail = body.detail;
        }
      } catch {
        // Ignore JSON parse failure and use default message.
      }
      throw new Error(detail);
    }

    const data = (await response.json()) as CreateSessionResponse;

    setActiveSession({
      ...config,
      sessionId: data.session_id,
      wsUrl: data.ws_url,
      wsPath: data.ws_path,
      apiBaseUrl: API_BASE_URL,
    });
  };

  const handleExitSession = () => {
    setActiveSession(null);
  };

  return (
    <div className="antialiased text-slate-200 selection:bg-cyan-500/30">
      {!activeSession ? (
        <SetupScreen onStart={handleStartSession} />
      ) : (
        <WorldViewport config={activeSession} onExit={handleExitSession} />
      )}
    </div>
  );
};

export default App;
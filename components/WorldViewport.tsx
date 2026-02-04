import React, { useEffect, useRef, useState, useCallback } from 'react';
import { ConnectionStatus, InputState, QualityProfile, SessionConfig, Telemetry } from '../types';
import { WEBSOCKET_RATE_MS, MOCK_LATENCY_BASE } from '../constants';
import { HUD } from './HUD';
import { Button } from './Button';

interface WorldViewportProps {
  config: SessionConfig;
  onExit: () => void;
}

export const WorldViewport: React.FC<WorldViewportProps> = ({ config, onExit }) => {
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.CONNECTING);
  const [telemetry, setTelemetry] = useState<Telemetry>({ fps: 0, bufferMs: 0, latencyMs: 0, generationTimeMs: 0 });
  const [latencyHistory, setLatencyHistory] = useState<{timestamp: number, value: number}[]>([]);
  
  // Refs for state that changes too fast for React state
  const inputState = useRef<InputState>({
    w: false, a: false, s: false, d: false, space: false, mouseX: 0, mouseY: 0
  });
  
  // Simulation State
  const cameraPos = useRef({ x: 0, y: 0, z: 0, yaw: 0, pitch: 0 });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number>();
  const lastTimeRef = useRef<number>(0);
  const wsIntervalRef = useRef<number>();

  // Input Event Handlers
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (key === 'w') inputState.current.w = true;
      if (key === 'a') inputState.current.a = true;
      if (key === 's') inputState.current.s = true;
      if (key === 'd') inputState.current.d = true;
      if (key === ' ') inputState.current.space = !inputState.current.space; // toggle
      if (key === 'r') {
         cameraPos.current = { x: 0, y: 0, z: 0, yaw: 0, pitch: 0 };
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if (key === 'w') inputState.current.w = false;
      if (key === 'a') inputState.current.a = false;
      if (key === 's') inputState.current.s = false;
      if (key === 'd') inputState.current.d = false;
    };

    const handleMouseMove = (e: MouseEvent) => {
      inputState.current.mouseX = e.movementX;
      inputState.current.mouseY = e.movementY;
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('mousemove', handleMouseMove);

    const canvas = canvasRef.current;
    const handleCanvasClick = () => {
      canvas?.requestPointerLock();
    };
    canvas?.addEventListener('click', handleCanvasClick);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('mousemove', handleMouseMove);
      canvas?.removeEventListener('click', handleCanvasClick);
    };
  }, []);

  // "WebSocket" Loop
  useEffect(() => {
    const connectTimer = setTimeout(() => {
      setStatus(ConnectionStatus.CONNECTED);
    }, 1500);

    wsIntervalRef.current = window.setInterval(() => {
      inputState.current.mouseX = 0;
      inputState.current.mouseY = 0;

      setTelemetry(prev => {
        const load = (inputState.current.w || inputState.current.a || inputState.current.s || inputState.current.d) ? 1.2 : 0.8;
        const newLat = MOCK_LATENCY_BASE * load + (Math.random() * 20 - 10);
        
        return {
          fps: 16 + (Math.random() * 2 - 1),
          bufferMs: 800 + (Math.random() * 100 - 50),
          latencyMs: newLat,
          generationTimeMs: 55 + (Math.random() * 5),
        };
      });

      setLatencyHistory(prev => {
        const newHistory = [...prev, { timestamp: Date.now(), value: MOCK_LATENCY_BASE + (Math.random() * 20) }];
        if (newHistory.length > 50) newHistory.shift();
        return newHistory;
      });

    }, WEBSOCKET_RATE_MS);

    return () => {
      clearTimeout(connectTimer);
      clearInterval(wsIntervalRef.current);
    };
  }, []);

  // Client-Side Rendering Loop
  const animate = useCallback((time: number) => {
    if (!lastTimeRef.current) lastTimeRef.current = time;
    const deltaTime = (time - lastTimeRef.current) / 1000;
    lastTimeRef.current = time;

    const speed = 5 * deltaTime;
    
    // Simple movement logic
    if (inputState.current.w) {
      cameraPos.current.z += Math.cos(cameraPos.current.yaw) * speed;
      cameraPos.current.x += Math.sin(cameraPos.current.yaw) * speed;
    }
    if (inputState.current.s) {
      cameraPos.current.z -= Math.cos(cameraPos.current.yaw) * speed;
      cameraPos.current.x -= Math.sin(cameraPos.current.yaw) * speed;
    }
    if (inputState.current.a) {
      cameraPos.current.x -= Math.cos(cameraPos.current.yaw) * speed;
      cameraPos.current.z += Math.sin(cameraPos.current.yaw) * speed;
    }
    if (inputState.current.d) {
      cameraPos.current.x += Math.cos(cameraPos.current.yaw) * speed;
      cameraPos.current.z -= Math.sin(cameraPos.current.yaw) * speed;
    }

    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        // Clear background
        ctx.fillStyle = '#0a0a0b'; // Zinc 950
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw Grid
        ctx.strokeStyle = '#27272a'; // Zinc 800
        ctx.lineWidth = 1;
        
        const gridSize = 50;
        const width = canvas.width;
        const height = canvas.height;
        const offsetX = (cameraPos.current.x * 100) % gridSize;
        const offsetY = (cameraPos.current.z * 100) % gridSize;

        ctx.save();
        ctx.translate(width/2, height/2);
        
        if (inputState.current.w || inputState.current.a || inputState.current.s || inputState.current.d) {
           ctx.translate(Math.random() * 1 - 0.5, Math.random() * 1 - 0.5);
        }
        
        // Floor grid
        ctx.beginPath();
        for(let i = -10; i <= 10; i++) {
            // Vertical (Z)
            ctx.moveTo(i * gridSize - offsetX, -height/2);
            ctx.lineTo((i * gridSize - offsetX) * 4, height/2);
            
            // Horizontal (X)
            ctx.moveTo(-width/2, i * gridSize + offsetY);
            ctx.lineTo(width/2, i * gridSize + offsetY);
        }
        ctx.stroke();

        // Text
        ctx.fillStyle = '#52525b'; // Zinc 600
        ctx.font = '12px Inter, sans-serif';
        ctx.fillText("Rendering Geometry...", -60, -20);
        
        ctx.fillStyle = '#3f3f46'; // Zinc 700
        ctx.font = '10px Inter, sans-serif';
        const promptPreview = config.prompt.length > 50 ? config.prompt.substring(0, 50) + "..." : config.prompt;
        ctx.fillText(promptPreview, -100, 0);

        ctx.restore();
      }
    }

    requestRef.current = requestAnimationFrame(animate);
  }, [config.prompt]);

  useEffect(() => {
    requestRef.current = requestAnimationFrame(animate);
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [animate]);

  return (
    <div className="relative w-full h-screen bg-black overflow-hidden flex flex-col items-center justify-center">
      
      {/* Main Stream View */}
      <div className="relative w-full h-full bg-black flex items-center justify-center">
        <canvas 
          ref={canvasRef}
          width={config.resolution === '720p' ? 1280 : 854}
          height={config.resolution === '720p' ? 720 : 480}
          className="max-w-full max-h-full object-contain"
        />
        
        {/* Connection Overlay */}
        {status !== ConnectionStatus.CONNECTED && (
          <div className="absolute inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-30">
             <div className="flex flex-col items-center gap-6">
               <div className="h-0.5 w-48 bg-zinc-800 rounded overflow-hidden">
                 <div className="h-full bg-white animate-[width_1.5s_ease-in-out_infinite]" style={{width: '30%'}}></div>
               </div>
               <div className="text-zinc-400 font-sans text-xs tracking-widest uppercase">Initializing Stream</div>
             </div>
          </div>
        )}

        <HUD 
          status={status}
          telemetry={telemetry}
          quality={config.quality}
          resolution={config.resolution}
          latencyHistory={latencyHistory}
        />
      </div>

      {/* Exit Button */}
      <div className="absolute top-6 right-6 z-50">
        <Button variant="secondary" onClick={onExit} className="text-xs py-1.5 px-4 bg-black/50 backdrop-blur-sm border-zinc-800">
          Disconnect
        </Button>
      </div>
    </div>
  );
};
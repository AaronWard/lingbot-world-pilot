import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActiveSession,
  ConnectionStatus,
  FrameHeader,
  InputState,
  Telemetry,
  TelemetryMessage,
} from '../types';
import { HUD } from './HUD';
import { Button } from './Button';
import { INPUT_SEND_RATE_MS, MAX_LATENCY_HISTORY } from '../constants';

interface WorldViewportProps {
  config: ActiveSession;
  onExit: () => void;
}

interface ParsedFramePacket {
  header: FrameHeader;
  jpegBytes: Uint8Array;
}

function parseFramePacket(buffer: ArrayBuffer): ParsedFramePacket {
  if (buffer.byteLength < 4) {
    throw new Error('Frame packet too small.');
  }

  const view = new DataView(buffer);
  const headerLength = view.getUint32(0, true);

  if (buffer.byteLength < 4 + headerLength) {
    throw new Error('Invalid frame packet header length.');
  }

  const headerBytes = new Uint8Array(buffer, 4, headerLength);
  const headerJson = new TextDecoder().decode(headerBytes);
  const header = JSON.parse(headerJson) as FrameHeader;
  const jpegBytes = new Uint8Array(buffer, 4 + headerLength);

  return { header, jpegBytes };
}

async function decodeJpegToImageSource(
  jpegBytes: Uint8Array
): Promise<ImageBitmap | HTMLImageElement> {
  const blob = new Blob([jpegBytes], { type: 'image/jpeg' });

  if ('createImageBitmap' in window) {
    return await createImageBitmap(blob);
  }

  return await new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(blob);

    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Failed to decode JPEG frame.'));
    };

    img.src = url;
  });
}

export const WorldViewport: React.FC<WorldViewportProps> = ({ config, onExit }) => {
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.CONNECTING);
  const [telemetry, setTelemetry] = useState<Telemetry>({
    fps: 0,
    bufferMs: 0,
    latencyMs: 0,
    generationTimeMs: 0,
  });
  const [latencyHistory, setLatencyHistory] = useState<{ timestamp: number; value: number }[]>([]);
  const [disconnecting, setDisconnecting] = useState(false);
  const [hasReceivedFrame, setHasReceivedFrame] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const inputSendIntervalRef = useRef<number | null>(null);
  const frameDrawTokenRef = useRef(0);

  const inputState = useRef<InputState>({
    w: false,
    a: false,
    s: false,
    d: false,
    space: false,
    mouseX: 0,
    mouseY: 0,
  });

  const inputSeqRef = useRef(0);

  const drawPlaceholder = useCallback(
    (title: string, subtitle?: string) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#050505';
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = '#1f1f22';
      ctx.lineWidth = 1;

      const horizonY = Math.floor(height * 0.42);
      const vanishX = width / 2;
      const gridHalfWidth = Math.floor(width * 0.48);

      for (let x = -gridHalfWidth; x <= gridHalfWidth; x += 50) {
        ctx.beginPath();
        ctx.moveTo(vanishX + x * 0.35, horizonY);
        ctx.lineTo(vanishX + x, height);
        ctx.stroke();
      }

      for (let y = horizonY; y <= height; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      ctx.fillStyle = '#71717a';
      ctx.font = '600 24px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(title, width / 2, height / 2 - 10);

      if (subtitle) {
        ctx.fillStyle = '#52525b';
        ctx.font = '14px Inter, sans-serif';
        ctx.fillText(subtitle, width / 2, height / 2 + 22);
      }
    },
    []
  );

  const drawFrame = useCallback(async (buffer: ArrayBuffer) => {
    const currentToken = ++frameDrawTokenRef.current;
    const { header, jpegBytes } = parseFramePacket(buffer);
    const imageSource = await decodeJpegToImageSource(jpegBytes);

    if (currentToken !== frameDrawTokenRef.current) {
      if ('close' in imageSource && typeof imageSource.close === 'function') {
        imageSource.close();
      }
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) {
      if ('close' in imageSource && typeof imageSource.close === 'function') {
        imageSource.close();
      }
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      if ('close' in imageSource && typeof imageSource.close === 'function') {
        imageSource.close();
      }
      return;
    }

    if (canvas.width !== header.w || canvas.height !== header.h) {
      canvas.width = header.w;
      canvas.height = header.h;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imageSource, 0, 0, canvas.width, canvas.height);

    if ('close' in imageSource && typeof imageSource.close === 'function') {
      imageSource.close();
    }

    setHasReceivedFrame(true);
  }, []);

  useEffect(() => {
    drawPlaceholder('Initializing Stream', 'Waiting for backend connection...');
  }, [drawPlaceholder]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();

      if (key === 'w') inputState.current.w = true;
      if (key === 'a') inputState.current.a = true;
      if (key === 's') inputState.current.s = true;
      if (key === 'd') inputState.current.d = true;

      if (key === ' ' && !e.repeat) {
        e.preventDefault();
        inputState.current.space = !inputState.current.space;
      }

      if (key === 'escape' && document.pointerLockElement === canvasRef.current) {
        document.exitPointerLock();
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
      if (document.pointerLockElement !== canvasRef.current) {
        return;
      }

      inputState.current.mouseX += e.movementX;
      inputState.current.mouseY += e.movementY;
    };

    const canvas = canvasRef.current;
    const handleCanvasClick = () => {
      canvas?.requestPointerLock();
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('mousemove', handleMouseMove);
    canvas?.addEventListener('click', handleCanvasClick);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('mousemove', handleMouseMove);
      canvas?.removeEventListener('click', handleCanvasClick);
    };
  }, []);

  useEffect(() => {
    const ws = new WebSocket(config.wsUrl);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    setStatus(ConnectionStatus.CONNECTING);
    setHasReceivedFrame(false);
    drawPlaceholder('Initializing Stream', 'Connecting to local renderer...');

    ws.onopen = () => {
      setStatus(ConnectionStatus.CONNECTED);
    };

    ws.onerror = () => {
      setStatus(ConnectionStatus.ERROR);
      if (!hasReceivedFrame) {
        drawPlaceholder('Stream Error', 'Failed to connect to renderer.');
      }
    };

    ws.onclose = () => {
      if (!disconnecting) {
        setStatus(ConnectionStatus.DISCONNECTED);
        if (!hasReceivedFrame) {
          drawPlaceholder('Disconnected', 'Session closed.');
        }
      }
    };

    ws.onmessage = async (event) => {
      try {
        if (typeof event.data === 'string') {
          const msg = JSON.parse(event.data) as TelemetryMessage;

          if (msg.type === 'telemetry') {
            setTelemetry({
              fps: msg.fps,
              bufferMs: msg.bufferMs,
              latencyMs: msg.latencyMs,
              generationTimeMs: msg.generationTimeMs,
            });

            setLatencyHistory((prev) => {
              const next = [
                ...prev,
                {
                  timestamp: Date.now(),
                  value: msg.latencyMs,
                },
              ];
              return next.slice(-MAX_LATENCY_HISTORY);
            });
          }

          return;
        }

        const arrayBuffer =
          event.data instanceof ArrayBuffer
            ? event.data
            : event.data instanceof Blob
              ? await event.data.arrayBuffer()
              : null;

        if (!arrayBuffer) {
          return;
        }

        await drawFrame(arrayBuffer);
      } catch (err) {
        console.error('Failed to handle WebSocket message:', err);
      }
    };

    inputSendIntervalRef.current = window.setInterval(() => {
      if (ws.readyState !== WebSocket.OPEN) {
        return;
      }

      const seq = inputSeqRef.current++;
      const payload = {
        type: 'input',
        seq,
        client_ts_ms: Date.now(),
        state: {
          ...inputState.current,
        },
      };

      ws.send(JSON.stringify(payload));

      inputState.current.mouseX = 0;
      inputState.current.mouseY = 0;
    }, INPUT_SEND_RATE_MS);

    return () => {
      if (inputSendIntervalRef.current !== null) {
        clearInterval(inputSendIntervalRef.current);
      }

      wsRef.current = null;

      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    };
  }, [config.wsUrl, disconnecting, drawFrame, drawPlaceholder, hasReceivedFrame]);

  const handleDisconnect = useCallback(async () => {
    setDisconnecting(true);

    try {
      const ws = wsRef.current;
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close();
      }

      await fetch(`${config.apiBaseUrl}/api/session/${config.sessionId}`, {
        method: 'DELETE',
      });
    } catch (err) {
      console.error('Failed to stop session cleanly:', err);
    } finally {
      onExit();
    }
  }, [config.apiBaseUrl, config.sessionId, onExit]);

  return (
    <div className="relative w-full h-screen bg-black overflow-hidden flex flex-col items-center justify-center">
      <div className="relative w-full h-full bg-black flex items-center justify-center">
        <canvas
          ref={canvasRef}
          width={config.resolution === '720p' ? 1280 : 832}
          height={config.resolution === '720p' ? 720 : 480}
          className="max-w-full max-h-full object-contain"
        />

        {status !== ConnectionStatus.CONNECTED && !hasReceivedFrame && (
          <div className="absolute inset-0 bg-black/50 backdrop-blur-[1px] flex items-center justify-center z-30">
            <div className="flex flex-col items-center gap-6">
              <div className="h-0.5 w-48 bg-zinc-800 rounded overflow-hidden">
                <div
                  className="h-full bg-white animate-[width_1.5s_ease-in-out_infinite]"
                  style={{ width: '30%' }}
                />
              </div>
              <div className="text-zinc-400 font-sans text-xs tracking-widest uppercase">
                {status === ConnectionStatus.ERROR ? 'Renderer Error' : 'Initializing Stream'}
              </div>
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

      <div className="absolute top-6 right-6 z-50">
        <Button
          variant="secondary"
          onClick={handleDisconnect}
          isLoading={disconnecting}
          className="text-xs py-1.5 px-4 bg-black/50 backdrop-blur-sm border-zinc-800"
        >
          Disconnect
        </Button>
      </div>
    </div>
  );
};
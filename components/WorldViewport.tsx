// components/WorldViewport.tsx
// ---------------------------------------------------------------------------
// Streams frames from the LingBot-World-Fast server. Matches App.tsx's
// contract: named export, props { config, onExit }. The session was already
// created via POST /api/session (which uploaded the init image), so here we
// just open the websocket the API returned and send {type:"init",session_id}.
//
//   text out: {type:"init",session_id}  {type:"input",keys[],dx,dy,dt}  {type:"stop"}
//   text in : {type:"ready",width,height,fps} | {type:"error",message}
//   bin  in : [u32 frameIndex][u16 w][u16 h][u32 jpegLen] + JPEG bytes
// ---------------------------------------------------------------------------
import React, { useEffect, useRef, useState } from "react";
import { ActiveSession, ConnectionStatus, Telemetry } from "../types";
import { HUD } from "./HUD";

interface WorldViewportProps {
  config: ActiveSession;
  onExit: () => void;
}

const MOVE_KEYS = new Set(["w", "a", "s", "d"]);
const INPUT_HZ = 30;

export const WorldViewport: React.FC<WorldViewportProps> = ({ config, onExit }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const heldKeys = useRef<Set<string>>(new Set());
  const mouse = useRef({ dx: 0, dy: 0 });
  const lastTick = useRef<number>(performance.now());
  const frameTimes = useRef<number[]>([]);   // recent frame arrival timestamps

  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.CONNECTING);
  const [pointerLocked, setPointerLocked] = useState(false);
  const [telemetry, setTelemetry] = useState<Telemetry>(
    { fps: 0, bufferMs: 0, latencyMs: 0, generationTimeMs: 0 } as Telemetry);
  const [latencyHistory, setLatencyHistory] = useState<{ timestamp: number; value: number }[]>([]);
  const [resolution, setResolution] = useState<string>(config.resolution);

  // --- websocket lifecycle -------------------------------------------------
  useEffect(() => {
    const url = `${config.wsUrl}${config.wsPath}`;
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "init", session_id: config.sessionId }));
    };

    ws.onmessage = async (ev) => {
      if (typeof ev.data === "string") {
        const msg = JSON.parse(ev.data);
        if (msg.type === "ready") {
          const c = canvasRef.current;
          if (c) {
            c.width = msg.width;
            c.height = msg.height;
            ctxRef.current = c.getContext("2d");
          }
          setResolution(`${msg.width}x${msg.height}`);
          setStatus(ConnectionStatus.CONNECTED);
        } else if (msg.type === "error") {
          setStatus(ConnectionStatus.ERROR);
          console.error("[world] server error:", msg.message);
        }
        return;
      }
      // binary frame
      const buf = ev.data as ArrayBuffer;
      const dv = new DataView(buf);
      const w = dv.getUint16(4, true);
      const h = dv.getUint16(6, true);
      const jpegLen = dv.getUint32(8, true);
      const jpeg = new Uint8Array(buf, 12, jpegLen);
      const bmp = await createImageBitmap(new Blob([jpeg], { type: "image/jpeg" }));
      const ctx = ctxRef.current;
      if (ctx) ctx.drawImage(bmp, 0, 0, w, h);
      bmp.close();

      // crude fps from frame arrival times over the last second
      const now = performance.now();
      const t = frameTimes.current;
      t.push(now);
      while (t.length && now - t[0] > 1000) t.shift();
      const fps = t.length;
      setTelemetry((prev) => ({ ...prev, fps } as Telemetry));
    };

    ws.onerror = () => setStatus(ConnectionStatus.ERROR);
    ws.onclose = () => setStatus((s) =>
      s === ConnectionStatus.CONNECTED ? ConnectionStatus.ERROR : s);

    return () => {
      try { ws.send(JSON.stringify({ type: "stop" })); } catch { /* ignore */ }
      ws.close();
      wsRef.current = null;
    };
  }, [config.wsUrl, config.wsPath, config.sessionId]);

  // --- keyboard (WASD + Esc to exit when not locked) -----------------------
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (MOVE_KEYS.has(k)) { heldKeys.current.add(k); e.preventDefault(); }
    };
    const up = (e: KeyboardEvent) => heldKeys.current.delete(e.key.toLowerCase());
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  // --- mouse look via pointer lock ----------------------------------------
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (document.pointerLockElement === canvasRef.current) {
        mouse.current.dx += e.movementX;
        mouse.current.dy += e.movementY;
      }
    };
    const onLockChange = () =>
      setPointerLocked(document.pointerLockElement === canvasRef.current);
    document.addEventListener("mousemove", onMove);
    document.addEventListener("pointerlockchange", onLockChange);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("pointerlockchange", onLockChange);
    };
  }, []);

  // --- input tick ----------------------------------------------------------
  useEffect(() => {
    if (status !== ConnectionStatus.CONNECTED) return;
    const id = setInterval(() => {
      const now = performance.now();
      const dt = (now - lastTick.current) / 1000;
      lastTick.current = now;
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify({
        type: "input",
        keys: Array.from(heldKeys.current),
        dx: mouse.current.dx,
        dy: mouse.current.dy,
        dt,
      }));
      mouse.current.dx = 0;
      mouse.current.dy = 0;
    }, 1000 / INPUT_HZ);
    return () => clearInterval(id);
  }, [status]);

  const handleExit = () => {
    try { wsRef.current?.send(JSON.stringify({ type: "stop" })); } catch { /* ignore */ }
    wsRef.current?.close();
    onExit();
  };

  return (
    <div className="relative w-screen h-screen bg-black overflow-hidden">
      <canvas
        ref={canvasRef}
        onClick={() => canvasRef.current?.requestPointerLock()}
        className="absolute inset-0 w-full h-full object-contain"
        style={{ cursor: pointerLocked ? "none" : "pointer" }}
      />

      <HUD
        status={status}
        telemetry={telemetry}
        quality={config.quality}
        resolution={resolution}
        latencyHistory={latencyHistory}
      />

      <button
        onClick={handleExit}
        className="absolute top-6 right-6 z-30 px-3 py-1.5 text-[10px] font-medium tracking-widest uppercase text-zinc-300 bg-black/50 backdrop-blur-sm border border-zinc-800/50 rounded-md hover:text-white hover:border-zinc-600 transition-colors"
      >
        Exit
      </button>

      {status !== ConnectionStatus.CONNECTED && (
        <div className="absolute inset-0 z-20 flex items-center justify-center text-zinc-300 text-sm pointer-events-none">
          {status === ConnectionStatus.CONNECTING && "Connecting to neural renderer…"}
          {status === ConnectionStatus.ERROR && "Connection error — check the server log."}
        </div>
      )}

      {status === ConnectionStatus.CONNECTED && !pointerLocked && (
        <div className="absolute inset-0 z-10 flex items-center justify-center text-zinc-400 text-xs pointer-events-none">
          Click to capture mouse · WASD to move · Esc to release
        </div>
      )}
    </div>
  );
};

export default WorldViewport;
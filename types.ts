export enum QualityProfile {
  LATENCY = 'latency',
  BALANCED = 'balanced',
  QUALITY = 'quality',
}

export enum ConnectionStatus {
  DISCONNECTED = 'DISCONNECTED',
  CONNECTING = 'CONNECTING',
  CONNECTED = 'CONNECTED',
  ERROR = 'ERROR',
}

export interface SessionConfig {
  prompt: string;
  resolution: '480p' | '720p';
  quality: QualityProfile;
  initImage: File | null;
}

export interface CreateSessionResponse {
  session_id: string;
  ws_url: string;
  ws_path: string;
  resolution: '480p' | '720p';
  quality: QualityProfile;
}

export interface ActiveSession extends SessionConfig {
  sessionId: string;
  wsUrl: string;
  wsPath: string;
  apiBaseUrl: string;
}

export interface InputState {
  w: boolean;
  a: boolean;
  s: boolean;
  d: boolean;
  space: boolean;
  mouseX: number;
  mouseY: number;
}

export interface Telemetry {
  fps: number;
  bufferMs: number;
  latencyMs: number;
  generationTimeMs: number;
}

export interface TelemetryMessage extends Telemetry {
  type: 'telemetry';
  server_ts_ms: number;
  lastInputSeq: number;
  lastInputClientTsMs: number;
}

export interface FrameHeader {
  type: 'frame';
  session_id: string;
  frame_id: number;
  chunk_id: number;
  chunk_frame_idx: number;
  w: number;
  h: number;
  format: 'jpeg';
  server_ts_ms: number;
  input_seq: number;
  input_client_ts_ms: number;
}
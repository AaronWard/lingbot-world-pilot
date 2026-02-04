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

export interface InputState {
  w: boolean;
  a: boolean;
  s: boolean;
  d: boolean;
  space: boolean; // idle toggle
  mouseX: number;
  mouseY: number;
}

export interface Telemetry {
  fps: number;
  bufferMs: number;
  latencyMs: number;
  generationTimeMs: number;
}
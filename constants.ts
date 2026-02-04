import { QualityProfile } from "./types";

export const DEFAULT_PROMPT = "A futuristic cyberpunk city street at night, neon lights, rain on wet pavement, low angle view.";

export const DEFAULT_CONFIG = {
  prompt: DEFAULT_PROMPT,
  resolution: '480p' as const,
  quality: QualityProfile.BALANCED,
  initImage: null,
};

// Simulated backend constants
export const WEBSOCKET_RATE_MS = 33; // ~30Hz
export const MOCK_LATENCY_BASE = 80; // ms

import { QualityProfile } from './types';

export const DEFAULT_PROMPT =
  'A serene forest with fog and beautiful sunlight breaking through the tree-line';

export const DEFAULT_CONFIG = {
  prompt: DEFAULT_PROMPT,
  resolution: '480p' as const,
  quality: QualityProfile.BALANCED,
  initImage: null,
};

const envApiBase = import.meta.env.VITE_LINGBOT_API_BASE as string | undefined;

export const API_BASE_URL =
  envApiBase?.replace(/\/$/, '') ??
  `${window.location.protocol}//${window.location.hostname}:8000`;

export const INPUT_SEND_RATE_MS = 50;
export const MAX_LATENCY_HISTORY = 50;
export const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') || 'http://localhost:8000';

export type PredictionResponse = {
  anomaly_score: number;
  threshold: number;
  label: 'ATTACK' | 'NORMAL';
  attack_type: string | null;
  confidence: number;
  is_attack: boolean;
  processing_time_ms: number;
  details?: Record<string, unknown> | null;
  decision?: 'attack' | 'normal';
  stage?: 'rule' | 'ml' | 'legacy';
  rule_id?: string | null;
  reason?: string | null;
  severity?: 'High' | 'Medium' | null;
  simulated?: boolean;
  top_features?: Array<{ feature: string; value: number; shap_value: number }>;
  can_id?: number;
  can_id_hex?: string;
  timestamp?: number | string;
};

export type LivePayload = {
  score: number;
  label: 'ATTACK' | 'NORMAL';
  attack_type?: string | null;
  confidence?: number;
  threshold: number;
  timestamp: string;
  is_attack?: boolean;
  pipeline_path?: string;
  rules_triggered?: unknown[];
  // Unified pipeline fields
  decision?: 'attack' | 'normal';
  stage?: 'rule' | 'ml' | 'legacy';
  rule_id?: string | null;
  reason?: string | null;
  severity?: 'High' | 'Medium' | null;
  simulated?: boolean;
  top_features?: Array<{ feature: string; value: number; shap_value: number }>;
};


export type HealthResponse = {
  status: string;
  pipeline_loaded: boolean;
  model: string;
  model_name: string;
  rules_count: number;
  shap_available: boolean;
  device: string;
  version?: string;
};

export type StatsResponse = {
  total_analyzed: number;
  attacks_detected: number;
  normals_detected: number;
  detection_rate: number;
  average_anomaly_score: number;
  attack_type_breakdown: Record<string, number>;
  last_prediction: {
    timestamp?: string;
    label?: string;
    attack_type?: string | null;
    score?: number;
    confidence?: number;
  };
  uptime_seconds: number;
};

export type ReportsManifest = {
  available_files: Array<{ name: string; url: string; size_bytes: number }>;
  metrics?: {
    metrics?: {
      accuracy?: number;
      precision?: number;
      recall?: number;
      f1?: number;
      auc?: number;
      ap?: number;
    };
  };
};

export type ThresholdConfig = {
  threshold: number;
  [key: string]: unknown;
};

export type ThresholdUpdateResponse = {
  threshold: number;
  persisted: boolean;
  threshold_data: Record<string, unknown>;
};

export type StreamConfigResponse = {
  interval_ms: number;
  frames_per_second: number;
};

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    const fallback = `Request failed with status ${response.status}`;
    let message = fallback;
    try {
      const data = await response.json();
      if (typeof data === 'string') {
        message = data;
      } else if (typeof data?.detail === 'string') {
        message = data.detail;
      } else if (Array.isArray(data?.detail) && data.detail.length > 0) {
        message = data.detail
          .map((item: any) => item.msg || item.message || JSON.stringify(item))
          .join('; ');
      } else if (data?.message) {
        message = data.message;
      }
    } catch {
      // JSON parse error, keep fallback
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

export const apiClient = {
  getHealth: () => requestJson<HealthResponse>('/health'),
  getStats: () => requestJson<StatsResponse>('/stats'),
  getThreshold: () => requestJson<ThresholdConfig>('/threshold'),
  updateThreshold: (threshold: number, persist = true) =>
    requestJson<ThresholdUpdateResponse>('/threshold', {
      method: 'POST',
      body: JSON.stringify({ threshold, persist }),
    }),
  getStreamConfig: () => requestJson<StreamConfigResponse>('/stream/config'),
  updateStreamConfig: (intervalMs: number) =>
    requestJson<StreamConfigResponse>('/stream/config', {
      method: 'POST',
      body: JSON.stringify({ interval_ms: intervalMs }),
    }),
  getVocabSize: () => requestJson<{ vocab_size: number }>('/vocab/size'),
  getReportsManifest: () => requestJson<ReportsManifest>('/reports/manifest'),
  predict: (sequence: string[], returnDetails = true) =>
    requestJson<PredictionResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify({ sequence, return_details: returnDetails }),
    }),
  predictBatch: (sequences: string[][], returnDetails = false) =>
    requestJson<{
      total: number;
      attacks: number;
      normals: number;
      average_score: number;
      predictions: PredictionResponse[];
    }>('/predict/batch', {
      method: 'POST',
      body: JSON.stringify({ sequences, return_details: returnDetails }),
    }),
  predictFrame: (frame: {
    timestamp: number;
    can_id: number | string;
    dlc?: number;
    data_bytes?: number[];
    simulated?: boolean;
  }) =>
    requestJson<PredictionResponse>('/predict/frame', {
      method: 'POST',
      body: JSON.stringify(frame),
    }),
  getReplayFrames: (className = 'Normal', start = 0, count = 50) =>
    requestJson<{
      class: string;
      start: number;
      count: number;
      total_test_frames: number;
      frames: Array<{
        timestamp: number;
        can_id: number;
        can_id_hex: string;
        dlc: number;
        data_bytes: number[];
        label: string;
        simulated: boolean;
      }>;
    }>(`/replay?class=${encodeURIComponent(className)}&start=${start}&count=${count}`),
  resetSession: (resetStats = false) =>
    requestJson<{ status: string; message: string }>(`/session/reset?reset_stats=${resetStats}`, {
      method: 'POST',
    }),
  getDemoFrames: (className = 'Normal', count = 20) =>
    requestJson<{
      class: string;
      count: number;
      total_test_frames: number;
      frames: Array<{
        timestamp: number;
        can_id: number;
        can_id_hex: string;
        dlc: number;
        data_bytes: number[];
        label: string;
        simulated: boolean;
      }>;
    }>(`/demo?class=${encodeURIComponent(className)}&count=${count}`),
  getReportAssetUrl: (fileName: string) => `${API_BASE}/static/reports/${fileName}`,
  getWebSocketUrl: () => API_BASE.replace(/^http/, 'ws') + '/ws/live',
};

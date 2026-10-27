// =============================================================================
// CAN BUS IDS DASHBOARD - TYPE DEFINITIONS
// =============================================================================

/**
 * WebSocket message types from the backend API
 */
export type MessageType = 'connection' | 'buffering' | 'prediction' | 'error';

/**
 * Connection status message from WebSocket
 */
export interface ConnectionMessage {
  type: 'connection';
  status: string;
  message: string;
  threshold: number;
  window_size: number;
}

/**
 * Buffering status message (before window is full)
 */
export interface BufferingMessage {
  type: 'buffering';
  buffer_size: number;
  frames_needed: number;
  can_id: string;
  is_unknown_id: boolean;
  message: string;
}

/**
 * Prediction result from the model (old /ws/predict format)
 */
export interface PredictionMessage {
  type: 'prediction';
  anomaly_score: number;
  is_attack: boolean;
  threshold: number;
  buffer_size: number;
  timestamp: string;
  can_id: string;
  confidence: number;
  is_unknown_id: boolean;
}

/**
 * Error message from WebSocket
 */
export interface ErrorMessage {
  type: 'error';
  error: string;
  details?: string;
  expected_format?: Record<string, string>;
}

/**
 * Union type for all possible WebSocket messages
 */
export type WebSocketMessage =
  | ConnectionMessage
  | BufferingMessage
  | PredictionMessage
  | ErrorMessage;

/**
 * Data point for the real-time chart
 */
export interface ChartDataPoint {
  timestamp: number;
  time: string;
  score: number;
  isAttack: boolean;
  canId: string;
}

/**
 * Severity levels for attack alerts
 */
export type Severity = 'High' | 'Medium' | null;

/**
 * SHAP feature importance entry
 */
export interface ShapFeature {
  feature: string;
  value: number;
  shap_value: number;
}

/**
 * Threat log entry — extended with hybrid pipeline fields
 */
export interface ThreatLogEntry {
  id: string;
  timestamp: string;
  localTime: string;
  anomalyScore: number;
  canId: string;
  isAttack: boolean;
  confidence: number;
  // Hybrid pipeline fields
  decision: 'attack' | 'normal';
  stage: 'rule' | 'ml' | 'legacy';
  attackType: string | null;
  ruleId: string | null;
  reason: string | null;
  severity: Severity;
  simulated: boolean;
  topFeatures: ShapFeature[];
}

/**
 * Dashboard statistics
 */
export interface DashboardStats {
  messagesProcessed: number;
  attacksDetected: number;
  currentScore: number;
  bufferSize: number;
  isConnected: boolean;
  networkStatus: 'secure' | 'threat' | 'buffering' | 'disconnected';
  threshold: number;
  lastUpdate: string;
  uptime: number;
}

/**
 * WebSocket connection state
 */
export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error';

/**
 * Custom hook return type for WebSocket
 */
export interface UseWebSocketReturn {
  connectionState: ConnectionState;
  stats: DashboardStats;
  chartData: ChartDataPoint[];
  threatLog: ThreatLogEntry[];
  lastPrediction: PredictionMessage | null;
  connect: () => void;
  disconnect: () => void;
  clearLog: () => void;
}

/**
 * CAN Frame to send to backend
 */
export interface CANFrame {
  can_id: string;
  timestamp?: number;
  dlc?: number;
  data?: string[];
}

/**
 * Live payload from /ws/live endpoint (hybrid pipeline unified schema)
 */
export interface LivePayload {
  score: number;
  label: 'ATTACK' | 'NORMAL';
  attack_type?: string | null;
  confidence?: number;
  threshold: number;
  timestamp: string;
  is_attack: boolean;
  pipeline_path?: string;
  rules_triggered?: unknown[];
  // New unified fields
  decision?: 'attack' | 'normal';
  stage?: 'rule' | 'ml' | 'legacy';
  rule_id?: string | null;
  reason?: string | null;
  severity?: Severity;
  simulated?: boolean;
  top_features?: ShapFeature[];
}

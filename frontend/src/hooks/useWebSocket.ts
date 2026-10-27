// =============================================================================
// useWebSocket.ts - Custom React Hook for WebSocket Connection
// Handles real-time streaming, reconnection logic, and state management
// =============================================================================

import { useState, useCallback, useRef, useEffect } from 'react';
import type {
  ConnectionState,
  DashboardStats,
  ChartDataPoint,
  ThreatLogEntry,
  PredictionMessage,
  UseWebSocketReturn,
  CANFrame,
  LivePayload,
} from '../types';
import { testDataGenerator } from '../utils/testDataGenerator';

// Configuration
const WS_URL = 'ws://localhost:8000/ws/live'; // Hybrid pipeline live stream
const MAX_CHART_POINTS = 100;      // Maximum data points to display
const MAX_LOG_ENTRIES = 50;        // Maximum threat log entries
const RECONNECT_DELAY = 3000;      // Reconnection delay in ms
const MAX_RECONNECT_ATTEMPTS = 10; // Maximum reconnection attempts
const TEST_DATA_INTERVAL = 100;    // Send frame every 100ms (10 frames/sec)
const ATTACK_PROBABILITY = 0.08;   // 8% chance of attack frame

/**
 * Custom hook for managing WebSocket connection to the CAN Bus IDS backend.
 * Provides real-time anomaly detection data, statistics, and automatic reconnection.
 */
export function useWebSocket(): UseWebSocketReturn {
  // Connection state
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const startTime = useRef<number>(Date.now());

  // Dashboard statistics
  const [stats, setStats] = useState<DashboardStats>({
    messagesProcessed: 0,
    attacksDetected: 0,
    currentScore: 0,
    bufferSize: 0,
    isConnected: false,
    networkStatus: 'disconnected',
    threshold: 3.26,
    lastUpdate: new Date().toISOString(),
    uptime: 0,
  });

  // Chart data (rolling window)
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);

  // Threat log
  const [threatLog, setThreatLog] = useState<ThreatLogEntry[]>([]);

  // Last prediction for display
  const [lastPrediction, setLastPrediction] = useState<PredictionMessage | null>(null);

  /**
   * Format timestamp for display
   */
  const formatTime = useCallback((isoString: string) => {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  }, []);

  /**
   * Handle incoming WebSocket messages from /ws/live (hybrid pipeline)
   */
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const raw = JSON.parse(event.data) as Record<string, unknown>;

      // Error frame
      if ('error' in raw) {
        console.error('[WS] Server error:', raw['error']);
        return;
      }

      // Unified /ws/live payload
      const msg = raw as unknown as LivePayload;
      const isAttack = msg.is_attack ?? (msg.label === 'ATTACK');
      const anomalyScore = msg.score ?? 0;

      // Update stats
      setStats(prev => ({
        ...prev,
        messagesProcessed: prev.messagesProcessed + 1,
        currentScore: anomalyScore,
        threshold: msg.threshold ?? prev.threshold,
        attacksDetected: isAttack ? prev.attacksDetected + 1 : prev.attacksDetected,
        networkStatus: isAttack ? 'threat' : 'secure',
        lastUpdate: msg.timestamp,
        uptime: Math.floor((Date.now() - startTime.current) / 1000),
        isConnected: true,
      }));

      // Chart data point
      const chartPoint: ChartDataPoint = {
        timestamp: Date.now(),
        time: formatTime(msg.timestamp),
        score: anomalyScore,
        isAttack,
        canId: '—',
      };
      setChartData(prev => {
        const next = [...prev, chartPoint];
        return next.slice(-MAX_CHART_POINTS);
      });

      // Threat log entry for attacks
      if (isAttack) {
        const logEntry: ThreatLogEntry = {
          id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          timestamp: msg.timestamp,
          localTime: formatTime(msg.timestamp),
          anomalyScore,
          canId: '—',
          isAttack: true,
          confidence: msg.confidence ?? 0,
          // Hybrid pipeline fields
          decision: msg.decision ?? 'attack',
          stage: (msg.stage as 'rule' | 'ml' | 'legacy') ?? 'legacy',
          attackType: msg.attack_type ?? null,
          ruleId: msg.rule_id ?? null,
          reason: msg.reason ?? null,
          severity: msg.severity ?? null,
          simulated: msg.simulated ?? false,
          topFeatures: msg.top_features ?? [],
        };
        setThreatLog(prev => [logEntry, ...prev].slice(0, MAX_LOG_ENTRIES));
      }

      // Also feed lastPrediction for components that need PredictionMessage shape
      setLastPrediction({
        type: 'prediction',
        anomaly_score: anomalyScore,
        is_attack: isAttack,
        threshold: msg.threshold ?? 0,
        buffer_size: 0,
        timestamp: msg.timestamp,
        can_id: '—',
        confidence: msg.confidence ?? 0,
        is_unknown_id: false,
      });
    } catch (error) {
      console.error('[WS] Failed to parse message:', error);
    }
  }, [formatTime]);

  /**
   * Send a CAN frame to the backend via WebSocket
   */
  const sendFrame = useCallback((frame: CANFrame) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(frame));
      } catch (error) {
        console.error('[WS] Failed to send frame:', error);
      }
    }
  }, []);

  /**
   * Connect to WebSocket server
   */
  const connect = useCallback(() => {
    // Don't connect if already connected
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    // Clear any existing reconnection timeout
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }

    setConnectionState('connecting');
    console.log('[WS] Connecting to', WS_URL);

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('[WS] Connection established');
        setConnectionState('connected');
        reconnectAttempts.current = 0;
        startTime.current = Date.now();

        // Start sending test CAN frames automatically
        console.log('[WS] Starting test data generator');
        testDataGenerator.start(sendFrame, TEST_DATA_INTERVAL, ATTACK_PROBABILITY);
      };

      ws.onmessage = handleMessage;

      ws.onerror = (error) => {
        console.error('[WS] Connection error:', error);
        setConnectionState('error');
      };

      ws.onclose = (event) => {
        console.log('[WS] Connection closed:', event.code, event.reason);
        setConnectionState('disconnected');
        setStats(prev => ({
          ...prev,
          isConnected: false,
          networkStatus: 'disconnected',
        }));

        // Stop test data generator
        testDataGenerator.stop();

        // Attempt to reconnect if not intentionally closed
        if (event.code !== 1000 && reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttempts.current++;
          console.log(`[WS] Reconnecting in ${RECONNECT_DELAY}ms (attempt ${reconnectAttempts.current}/${MAX_RECONNECT_ATTEMPTS})`);

          reconnectTimeout.current = setTimeout(() => {
            connect();
          }, RECONNECT_DELAY);
        }
      };
    } catch (error) {
      console.error('[WS] Failed to create WebSocket:', error);
      setConnectionState('error');
    }
  }, [handleMessage, sendFrame]);

  /**
   * Disconnect from WebSocket server
   */
  const disconnect = useCallback(() => {
    // Stop test data generator
    testDataGenerator.stop();

    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close(1000, 'User initiated disconnect');
      wsRef.current = null;
    }

    setConnectionState('disconnected');
    reconnectAttempts.current = 0;
  }, []);

  /**
   * Clear threat log
   */
  const clearLog = useCallback(() => {
    setThreatLog([]);
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    connect();

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // Update uptime periodically
  useEffect(() => {
    const interval = setInterval(() => {
      if (connectionState === 'connected') {
        setStats(prev => ({
          ...prev,
          uptime: Math.floor((Date.now() - startTime.current) / 1000),
        }));
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [connectionState]);

  return {
    connectionState,
    stats,
    chartData,
    threatLog,
    lastPrediction,
    connect,
    disconnect,
    clearLog,
  };
}

export default useWebSocket;

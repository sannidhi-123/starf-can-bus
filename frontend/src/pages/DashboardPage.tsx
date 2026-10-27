// FILE: frontend/src/pages/DashboardPage.tsx - STARF Hybrid Pipeline Dashboard
import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  Activity,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Database,
  Play,
  RotateCcw,
  Send,
  Shield,
  Sparkles,
  Square,
} from 'lucide-react';
import { apiClient, type PredictionResponse } from '../api/client';
import { AppLayout, useAnalysisControl } from '../layout/AppLayout';
import { AdvancedAnalytics, type FrameTelemetry } from '../components/AdvancedAnalytics';
import { testDataGenerator } from '../utils/testDataGenerator';

const MAX_SESSION_FRAMES = 120;
const MAX_FEED_ALERTS = 25;

const ATTACK_CLASSES = ['Normal', 'DoS', 'Fuzzy', 'Spoof_Gear', 'Spoof_RPM'] as const;
type AttackClass = (typeof ATTACK_CLASSES)[number];

const ATTACK_TYPE_COLORS: Record<string, string> = {
  'DoS': '#DC2626',
  'Fuzzy': '#D97706',
  'Spoof_Gear': '#2563EB',
  'Spoof_RPM': '#7C3AED',
  'Rule violation': '#EA580C',
  'Normal': '#10B981',
};

const CARD_STYLE: CSSProperties = {
  background: 'var(--color-bg-card)',
  border: '1px solid var(--color-border)',
  borderRadius: '10px',
  boxShadow: 'var(--shadow-card)',
};

const SECTION_LABEL_STYLE: CSSProperties = {
  fontSize: '0.68rem',
  fontWeight: 600,
  textTransform: 'uppercase',
  letterSpacing: '0.09em',
  color: 'var(--color-text-muted)',
  paddingLeft: '8px',
  borderLeft: '3px solid var(--color-primary)',
  marginBottom: '12px',
  lineHeight: 1,
};

export type SessionFrame = {
  id: number;
  time: string;
  timestamp: number;
  canId: string;
  canIdHex: string;
  dlc: number;
  dataBytes: number[];
  decision: 'attack' | 'normal';
  stage: 'rule' | 'ml' | 'legacy';
  attackType: string;
  severity: 'High' | 'Medium' | null;
  confidence: number;
  ruleId: string | null;
  reason: string | null;
  simulated: boolean;
  topFeatures: Array<{ feature: string; value: number; shap_value: number }>;
};

export function DashboardPage() {
  return (
    <AppLayout>
      <DashboardContent />
    </AppLayout>
  );
}

function DashboardContent() {
  const { isAnalysisRunning } = useAnalysisControl();

  // Traffic Source selection
  const [trafficSource, setTrafficSource] = useState<'replay' | 'simulated' | 'manual'>('replay');
  const [replayClass, setReplayClass] = useState<AttackClass>('DoS');
  const [replayIndex, setReplayIndex] = useState(0);
  const [replayTotalAvailable, setReplayTotalAvailable] = useState(20000);
  const [streamSpeedMs, setStreamSpeedMs] = useState(600);
  const [isReplaying, setIsReplaying] = useState(false);

  // Manual Frame form inputs
  const [manualCanId, setManualCanId] = useState('0x000');
  const [manualDlc, setManualDlc] = useState(8);
  const [manualDataBytes, setManualDataBytes] = useState('00 00 00 00 00 00 00 00');
  const [manualLoading, setManualLoading] = useState(false);
  const [manualNotice, setManualNotice] = useState<string | null>(null);

  // Session state (evaluated responses in this session)
  const [sessionFrames, setSessionFrames] = useState<SessionFrame[]>([]);
  const [expandedShapId, setExpandedShapId] = useState<number | null>(null);

  const frameIdCounter = useRef(0);
  const replayIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Helper to ingest an evaluated frame response into the session
  const ingestFrame = (
    result: PredictionResponse,
    raw: {
      can_id: number | string;
      can_id_hex?: string;
      dlc?: number;
      data_bytes?: number[];
      timestamp?: number;
      simulated?: boolean;
    }
  ) => {
    frameIdCounter.current += 1;
    const cidNum = typeof raw.can_id === 'number' ? raw.can_id : parseInt(String(raw.can_id), 16) || 0;
    const cidHex = raw.can_id_hex || `0x${cidNum.toString(16).toUpperCase().padStart(3, '0')}`;
    const dbytes = raw.data_bytes || [0, 0, 0, 0, 0, 0, 0, 0];
    const ts = raw.timestamp || Date.now() / 1000;
    const timeStr = new Date(ts * 1000).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

    const isAttack = result.decision ? result.decision === 'attack' : (result.is_attack ?? (result.label === 'ATTACK'));
    const stage = result.stage ?? (result.rule_id ? 'rule' : 'ml');
    const attackType = result.attack_type || (stage === 'rule' ? 'Rule violation' : (isAttack ? 'Attack' : 'Normal'));
    const confidence = typeof result.confidence === 'number' ? result.confidence : 0.0;
    const severity = result.severity ?? (isAttack ? (stage === 'rule' ? 'High' : 'Medium') : null);
    const simulated = Boolean(raw.simulated ?? result.simulated ?? false);

    const frame: SessionFrame = {
      id: frameIdCounter.current,
      time: timeStr,
      timestamp: ts,
      canId: String(cidNum),
      canIdHex: cidHex,
      dlc: raw.dlc ?? 8,
      dataBytes: dbytes,
      decision: isAttack ? 'attack' : 'normal',
      stage,
      attackType: isAttack ? attackType : 'Normal',
      severity,
      confidence,
      ruleId: result.rule_id ?? null,
      reason: result.reason ?? null,
      simulated,
      topFeatures: result.top_features ?? [],
    };

    setSessionFrames((prev) => [...prev, frame].slice(-MAX_SESSION_FRAMES));
  };

  // Replay chunk fetcher
  const stepReplay = async (count = 10) => {
    try {
      const data = await apiClient.getReplayFrames(replayClass, replayIndex, count);
      setReplayTotalAvailable(data.total_test_frames);
      setReplayIndex((prev) => (prev + data.frames.length) % Math.max(1, data.total_test_frames));

      for (const rawFrame of data.frames) {
        const res = await apiClient.predictFrame({
          timestamp: rawFrame.timestamp,
          can_id: rawFrame.can_id,
          dlc: rawFrame.dlc,
          data_bytes: rawFrame.data_bytes,
          simulated: false,
        });
        ingestFrame(res, rawFrame);
      }
    } catch (err) {
      console.error('Replay error:', err);
    }
  };

  // Continuous replay timer
  useEffect(() => {
    if (trafficSource === 'replay' && isReplaying && isAnalysisRunning) {
      replayIntervalRef.current = setInterval(() => {
        stepReplay(1);
      }, streamSpeedMs);
    } else {
      if (replayIntervalRef.current) {
        clearInterval(replayIntervalRef.current);
        replayIntervalRef.current = null;
      }
    }

    return () => {
      if (replayIntervalRef.current) {
        clearInterval(replayIntervalRef.current);
      }
    };
  }, [trafficSource, isReplaying, isAnalysisRunning, replayClass, replayIndex, streamSpeedMs]);

  // Initial replay sample on first mount so the user has immediate real data
  useEffect(() => {
    let active = true;
    apiClient
      .getReplayFrames('DoS', 0, 12)
      .then(async (data) => {
        if (!active) return;
        setReplayTotalAvailable(data.total_test_frames);
        setReplayIndex(12);
        for (const rawFrame of data.frames) {
          const res = await apiClient.predictFrame({
            timestamp: rawFrame.timestamp,
            can_id: rawFrame.can_id,
            dlc: rawFrame.dlc,
            data_bytes: rawFrame.data_bytes,
            simulated: false,
          });
          if (active) ingestFrame(res, rawFrame);
        }
      })
      .catch(() => undefined);

    return () => {
      active = false;
    };
  }, []);

  // Simulated traffic generator loop
  useEffect(() => {
    if (trafficSource === 'simulated' && isAnalysisRunning && isReplaying) {
      testDataGenerator.start(
        async (genFrame) => {
          try {
            const cidNum = parseInt(genFrame.can_id || '0', 16) || 0;
            const dbytes = (genFrame.data || []).map((b) => parseInt(b, 16) || 0);
            const ts = genFrame.timestamp ?? (Date.now() / 1000);
            const res = await apiClient.predictFrame({
              timestamp: ts,
              can_id: cidNum,
              dlc: genFrame.dlc ?? dbytes.length,
              data_bytes: dbytes,
              simulated: true,
            });
            ingestFrame(res, {
              timestamp: ts,
              can_id: cidNum,
              can_id_hex: `0x${(genFrame.can_id || '0').toUpperCase()}`,
              dlc: genFrame.dlc ?? dbytes.length,
              data_bytes: dbytes,
              simulated: true,
            });
          } catch (e) {
            console.error('Simulated inference error:', e);
          }
        },
        streamSpeedMs,
        0.25
      );
    } else {
      testDataGenerator.stop();
    }

    return () => {
      testDataGenerator.stop();
    };
  }, [trafficSource, isAnalysisRunning, isReplaying, streamSpeedMs]);

  // Handle Manual Frame Submission
  const handleManualSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setManualLoading(true);
    setManualNotice(null);

    try {
      const cid = manualCanId.startsWith('0x') || manualCanId.startsWith('0X')
        ? parseInt(manualCanId, 16)
        : parseInt(manualCanId, 10);
      const byteTokens = manualDataBytes.split(/[\s,]+/).filter(Boolean);
      const dbytes = byteTokens.map((t) => (t.startsWith('0x') ? parseInt(t, 16) : parseInt(t, 16) || 0)).slice(0, 8);
      while (dbytes.length < 8) dbytes.push(0);

      const ts = Date.now() / 1000;
      const res = await apiClient.predictFrame({
        timestamp: ts,
        can_id: cid,
        dlc: manualDlc,
        data_bytes: dbytes,
        simulated: false,
      });

      ingestFrame(res, {
        timestamp: ts,
        can_id: cid,
        can_id_hex: `0x${cid.toString(16).toUpperCase().padStart(3, '0')}`,
        dlc: manualDlc,
        data_bytes: dbytes,
        simulated: false,
      });

      setManualNotice(`Evaluated ${cid}: ${res.decision?.toUpperCase() ?? res.label} (${res.attack_type || (res.rule_id ? 'Rule violation' : 'Normal')})`);
    } catch (err: any) {
      setManualNotice(`Error: ${err.message || 'Submission failed'}`);
    } finally {
      setManualLoading(false);
    }
  };

  const handleResetSession = () => {
    setSessionFrames([]);
    frameIdCounter.current = 0;
  };

  // ---------------------------------------------------------------------------
  // Metrics Computed from Actual Responses in This Session
  // ---------------------------------------------------------------------------
  const messagesAnalyzed = sessionFrames.length;
  const alertsList = useMemo(() => sessionFrames.filter((f) => f.decision === 'attack'), [sessionFrames]);
  const alertsTriggered = alertsList.length;

  const detectionRate = useMemo(() => {
    if (messagesAnalyzed === 0) return 0;
    return (alertsTriggered / messagesAnalyzed) * 100;
  }, [alertsTriggered, messagesAnalyzed]);

  const meanConfidence = useMemo(() => {
    if (messagesAnalyzed === 0) return 0;
    const total = sessionFrames.reduce((acc, f) => acc + f.confidence, 0);
    return total / messagesAnalyzed;
  }, [sessionFrames, messagesAnalyzed]);

  // Stage split (Rule vs ML)
  const stageSplit = useMemo(() => {
    const rules = alertsList.filter((a) => a.stage === 'rule').length;
    const ml = alertsList.filter((a) => a.stage === 'ml').length;
    return { rules, ml };
  }, [alertsList]);

  // Attack breakdown counts
  const attackBreakdown = useMemo(() => {
    const counts: Record<string, number> = {
      'DoS': 0,
      'Fuzzy': 0,
      'Spoof_Gear': 0,
      'Spoof_RPM': 0,
      'Rule violation': 0,
    };
    alertsList.forEach((a) => {
      const type = a.attackType;
      if (type in counts) counts[type]++;
      else counts['Rule violation']++;
    });
    return counts;
  }, [alertsList]);

  // Severity breakdown
  const severityBreakdown = useMemo(() => {
    let high = 0;
    let medium = 0;
    alertsList.forEach((a) => {
      if (a.severity === 'High') high++;
      else if (a.severity === 'Medium') medium++;
      else if (a.stage === 'rule') high++;
      else medium++;
    });
    return { high, medium };
  }, [alertsList]);

  // Chart data for Detection Feed
  const chartData = useMemo(() => {
    if (sessionFrames.length === 0) {
      return [{ time: '--:--:--', confidence: 0, stage: 'ml', severity: null, decision: 'normal' }];
    }
    return sessionFrames.slice(-60).map((f) => ({
      time: f.time,
      confidence: Math.round(f.confidence * 100),
      stage: f.stage,
      severity: f.severity,
      decision: f.decision,
      attackType: f.attackType,
    }));
  }, [sessionFrames]);

  // Telemetry for Advanced Analytics components
  const telemetryFrames: FrameTelemetry[] = useMemo(() => {
    return sessionFrames.map((f) => ({
      timestamp: f.timestamp,
      canId: f.canId,
      canIdHex: f.canIdHex,
      dataBytes: f.dataBytes,
    }));
  }, [sessionFrames]);

  return (
    <div style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}>
      {/* Top Source Mode Banner */}
      <div
        style={{
          ...CARD_STYLE,
          padding: '12px 18px',
          marginBottom: '16px',
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px',
          background: 'linear-gradient(90deg, var(--color-bg-card) 0%, var(--color-primary-light) 100%)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span
            style={{
              padding: '4px 10px',
              borderRadius: '6px',
              fontSize: '0.72rem',
              fontWeight: 700,
              letterSpacing: '0.04em',
              textTransform: 'uppercase',
              background: 'var(--color-primary)',
              color: 'white',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <Database size={12} />
            Traffic Source: {trafficSource === 'replay' ? `Dataset Replay (${replayClass})` : trafficSource === 'simulated' ? 'Simulated Generator' : 'Manual Frame'}
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>
            {trafficSource === 'replay'
              ? `Real held-out test frames from data/prepared.parquet (Offset: ${replayIndex}/${replayTotalAvailable})`
              : trafficSource === 'simulated'
              ? 'TestDataGenerator active with simulated=true tags'
              : 'Single frame manual submission'}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            type="button"
            onClick={() => setTrafficSource('replay')}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
              border: trafficSource === 'replay' ? '1px solid var(--color-primary)' : '1px solid var(--color-border)',
              background: trafficSource === 'replay' ? 'var(--color-primary)' : 'var(--color-bg-base)',
              color: trafficSource === 'replay' ? 'white' : 'var(--color-text-secondary)',
            }}
          >
            Dataset Replay
          </button>
          <button
            type="button"
            onClick={() => setTrafficSource('simulated')}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
              border: trafficSource === 'simulated' ? '1px solid var(--color-primary)' : '1px solid var(--color-border)',
              background: trafficSource === 'simulated' ? 'var(--color-primary)' : 'var(--color-bg-base)',
              color: trafficSource === 'simulated' ? 'white' : 'var(--color-text-secondary)',
            }}
          >
            Simulated
          </button>
          <button
            type="button"
            onClick={() => setTrafficSource('manual')}
            style={{
              padding: '6px 12px',
              borderRadius: '6px',
              fontSize: '0.75rem',
              fontWeight: 600,
              cursor: 'pointer',
              border: trafficSource === 'manual' ? '1px solid var(--color-primary)' : '1px solid var(--color-border)',
              background: trafficSource === 'manual' ? 'var(--color-primary)' : 'var(--color-bg-base)',
              color: trafficSource === 'manual' ? 'white' : 'var(--color-text-secondary)',
            }}
          >
            Manual Frame
          </button>
          <button
            type="button"
            onClick={handleResetSession}
            title="Reset session telemetry"
            style={{
              padding: '6px 10px',
              borderRadius: '6px',
              border: '1px solid var(--color-border)',
              background: 'var(--color-bg-base)',
              color: 'var(--color-text-muted)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              fontSize: '0.72rem',
            }}
          >
            <RotateCcw size={12} />
            Reset
          </button>
        </div>
      </div>

      {/* Sub-Panel for Selected Source Controls */}
      {trafficSource === 'replay' && (
        <div style={{ ...CARD_STYLE, padding: '12px 18px', marginBottom: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>Target Attack Class:</span>
            <select
              value={replayClass}
              onChange={(e) => {
                setReplayClass(e.target.value as AttackClass);
                setReplayIndex(0);
              }}
              style={{
                fontSize: '0.78rem',
                padding: '5px 10px',
                borderRadius: '6px',
                border: '1px solid var(--color-border)',
                background: 'var(--color-bg-base)',
                color: 'var(--color-text-primary)',
                fontWeight: 600,
              }}
            >
              {ATTACK_CLASSES.map((cls) => (
                <option key={cls} value={cls}>
                  {cls}
                </option>
              ))}
            </select>

            <label style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
              Speed:
              <select
                value={streamSpeedMs}
                onChange={(e) => setStreamSpeedMs(Number(e.target.value))}
                style={{
                  fontSize: '0.72rem',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  border: '1px solid var(--color-border)',
                  background: 'var(--color-bg-base)',
                  color: 'var(--color-text-primary)',
                }}
              >
                <option value={100}>10 fps (100ms)</option>
                <option value={250}>4 fps (250ms)</option>
                <option value={500}>2 fps (500ms)</option>
                <option value={1000}>1 fps (1000ms)</option>
              </select>
            </label>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              type="button"
              onClick={() => stepReplay(20)}
              style={{
                padding: '6px 14px',
                borderRadius: '6px',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                background: 'var(--color-primary-light)',
                color: 'var(--color-primary)',
                border: '1px solid rgba(26,111,212,0.3)',
              }}
            >
              Step 20 Frames
            </button>
            <button
              type="button"
              onClick={() => setIsReplaying((prev) => !prev)}
              style={{
                padding: '6px 14px',
                borderRadius: '6px',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                background: isReplaying ? 'var(--color-danger-bg)' : 'var(--color-success-bg)',
                color: isReplaying ? 'var(--color-danger)' : 'var(--color-success)',
                border: isReplaying ? '1px solid rgba(196,30,58,0.3)' : '1px solid rgba(15,123,85,0.3)',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              {isReplaying ? <Square size={12} /> : <Play size={12} />}
              {isReplaying ? 'Pause Replay' : 'Continuous Stream'}
            </button>
          </div>
        </div>
      )}

      {trafficSource === 'simulated' && (
        <div style={{ ...CARD_STYLE, padding: '12px 18px', marginBottom: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)' }}>
              Synthetic stream generated client-side via <code>TestDataGenerator</code>. Every alert is stamped with <span style={{ color: 'var(--color-warning)', fontWeight: 700 }}>SIMULATED</span>.
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              type="button"
              onClick={() => testDataGenerator.injectAttackBurst(10, 50)}
              style={{
                padding: '6px 14px',
                borderRadius: '6px',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                background: 'var(--color-danger-bg)',
                color: 'var(--color-danger)',
                border: '1px solid rgba(196,30,58,0.3)',
              }}
            >
              Inject 10-Frame Attack Burst
            </button>
            <button
              type="button"
              onClick={() => setIsReplaying((prev) => !prev)}
              style={{
                padding: '6px 14px',
                borderRadius: '6px',
                fontSize: '0.75rem',
                fontWeight: 600,
                cursor: 'pointer',
                background: isReplaying ? 'var(--color-danger-bg)' : 'var(--color-primary-light)',
                color: isReplaying ? 'var(--color-danger)' : 'var(--color-primary)',
                border: '1px solid var(--color-border)',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              {isReplaying ? <Square size={12} /> : <Play size={12} />}
              {isReplaying ? 'Stop Generator' : 'Start Generator'}
            </button>
          </div>
        </div>
      )}

      {trafficSource === 'manual' && (
        <form onSubmit={handleManualSubmit} style={{ ...CARD_STYLE, padding: '14px 18px', marginBottom: '16px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '12px', alignItems: 'end' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.68rem', fontWeight: 600, color: 'var(--color-text-muted)', marginBottom: '4px' }}>
                CAN ID (Hex or Dec)
              </label>
              <input
                type="text"
                value={manualCanId}
                onChange={(e) => setManualCanId(e.target.value)}
                placeholder="0x000 or 0x18F"
                style={{
                  width: '100%',
                  padding: '6px 10px',
                  borderRadius: '6px',
                  border: '1px solid var(--color-border)',
                  background: 'var(--color-bg-base)',
                  fontSize: '0.8rem',
                  fontFamily: 'JetBrains Mono, monospace',
                  color: 'var(--color-text-primary)',
                }}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '0.68rem', fontWeight: 600, color: 'var(--color-text-muted)', marginBottom: '4px' }}>
                DLC (0-8)
              </label>
              <input
                type="number"
                min={0}
                max={8}
                value={manualDlc}
                onChange={(e) => setManualDlc(Number(e.target.value))}
                style={{
                  width: '100%',
                  padding: '6px 10px',
                  borderRadius: '6px',
                  border: '1px solid var(--color-border)',
                  background: 'var(--color-bg-base)',
                  fontSize: '0.8rem',
                  fontFamily: 'JetBrains Mono, monospace',
                  color: 'var(--color-text-primary)',
                }}
              />
            </div>
            <div style={{ gridColumn: 'span 2' }}>
              <label style={{ display: 'block', fontSize: '0.68rem', fontWeight: 600, color: 'var(--color-text-muted)', marginBottom: '4px' }}>
                Data Bytes (8 Hex values, e.g. 00 11 22 33 44 55 66 77)
              </label>
              <input
                type="text"
                value={manualDataBytes}
                onChange={(e) => setManualDataBytes(e.target.value)}
                placeholder="00 00 00 00 00 00 00 00"
                style={{
                  width: '100%',
                  padding: '6px 10px',
                  borderRadius: '6px',
                  border: '1px solid var(--color-border)',
                  background: 'var(--color-bg-base)',
                  fontSize: '0.8rem',
                  fontFamily: 'JetBrains Mono, monospace',
                  color: 'var(--color-text-primary)',
                }}
              />
            </div>
            <div>
              <button
                type="submit"
                disabled={manualLoading}
                style={{
                  width: '100%',
                  padding: '7px 16px',
                  borderRadius: '6px',
                  background: 'var(--color-primary)',
                  color: 'white',
                  fontWeight: 600,
                  fontSize: '0.8rem',
                  cursor: 'pointer',
                  border: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                }}
              >
                <Send size={14} />
                {manualLoading ? 'Analyzing...' : 'Analyze Frame'}
              </button>
            </div>
          </div>
          {manualNotice && (
            <div style={{ marginTop: '10px', fontSize: '0.75rem', color: 'var(--color-primary)', fontWeight: 600 }}>
              {manualNotice}
            </div>
          )}
        </form>
      )}

      {/* 4 Session Metric Cards */}
      <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4" style={{ gap: '12px' }}>
        <article style={{ ...CARD_STYLE, padding: '14px 16px', borderRadius: '10px' }}>
          <div style={{ width: '28px', height: '28px', borderRadius: '7px', background: 'var(--color-primary-light)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Activity size={14} color="var(--color-primary)" />
          </div>
          <div style={{ fontSize: '1.3rem', fontWeight: 700, fontFamily: 'JetBrains Mono, monospace', color: 'var(--color-text-primary)', marginTop: '8px', lineHeight: 1.1 }}>
            {messagesAnalyzed.toLocaleString()}
          </div>
          <div style={{ fontSize: '0.68rem', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.09em', color: 'var(--color-text-muted)', marginTop: '2px', lineHeight: 1 }}>
            Messages Analyzed
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginTop: '6px' }}>
            Session frames received
          </div>
        </article>

        <article style={{ ...CARD_STYLE, padding: '14px 16px', borderRadius: '10px' }}>
          <div style={{ width: '28px', height: '28px', borderRadius: '7px', background: 'rgba(196,30,58,0.12)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <AlertTriangle size={14} color="var(--color-danger)" />
          </div>
          <div style={{ fontSize: '1.3rem', fontWeight: 700, fontFamily: 'JetBrains Mono, monospace', color: 'var(--color-text-primary)', marginTop: '8px', lineHeight: 1.1 }}>
            {alertsTriggered.toLocaleString()}
          </div>
          <div style={{ fontSize: '0.68rem', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.09em', color: 'var(--color-text-muted)', marginTop: '2px', lineHeight: 1 }}>
            Alerts Triggered
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--color-danger)', marginTop: '6px' }}>
            {stageSplit.rules} rule · {stageSplit.ml} ML
          </div>
        </article>

        <article style={{ ...CARD_STYLE, padding: '14px 16px', borderRadius: '10px' }}>
          <div style={{ width: '28px', height: '28px', borderRadius: '7px', background: 'rgba(15,123,85,0.12)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Shield size={14} color="var(--color-success)" />
          </div>
          <div style={{ fontSize: '1.3rem', fontWeight: 700, fontFamily: 'JetBrains Mono, monospace', color: 'var(--color-text-primary)', marginTop: '8px', lineHeight: 1.1 }}>
            {detectionRate.toFixed(2)}%
          </div>
          <div style={{ fontSize: '0.68rem', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.09em', color: 'var(--color-text-muted)', marginTop: '2px', lineHeight: 1 }}>
            Detection Rate (Alerts / Analyzed)
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--color-success)', marginTop: '6px' }}>
            {alertsTriggered} / {messagesAnalyzed} frames
          </div>
        </article>

        <article style={{ ...CARD_STYLE, padding: '14px 16px', borderRadius: '10px' }}>
          <div style={{ width: '28px', height: '28px', borderRadius: '7px', background: 'rgba(124,58,237,0.12)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Sparkles size={14} color="#7C3AED" />
          </div>
          <div style={{ fontSize: '1.3rem', fontWeight: 700, fontFamily: 'JetBrains Mono, monospace', color: 'var(--color-text-primary)', marginTop: '8px', lineHeight: 1.1 }}>
            {(meanConfidence * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: '0.68rem', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.09em', color: 'var(--color-text-muted)', marginTop: '2px', lineHeight: 1 }}>
            Mean Confidence
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--color-text-secondary)', marginTop: '6px' }}>
            Average model & rule certainty
          </div>
        </article>
      </section>

      {/* Main Grid: Detection Feed + Attack Breakdown */}
      <section className="grid grid-cols-1 xl:grid-cols-12" style={{ gap: '16px', marginTop: '16px' }}>
        {/* Detection Feed (Plots Confidence by Stage & Severity) */}
        <article className="xl:col-span-8" style={{ ...CARD_STYLE, padding: '20px 24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' }}>
            <div>
              <div style={SECTION_LABEL_STYLE}>Detection Feed</div>
              <h2 style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                Real-Time Frame Confidence &amp; Detection Stage
              </h2>
              <p style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)', marginTop: '2px' }}>
                Frame-level classification confidence colored by decision stage (Rule vs ML) and severity.
              </p>
            </div>

            {/* Stage / Severity Legend */}
            <div style={{ display: 'flex', gap: '12px', fontSize: '0.7rem', color: 'var(--color-text-secondary)' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#DC2626' }} />
                Rule (High)
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#D97706' }} />
                Rule (Med)
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#2563EB' }} />
                ML Attack
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10B981' }} />
                Normal
              </span>
            </div>
          </div>

          <div style={{ marginTop: '12px', height: '280px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F4" />
                <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#8896A8' }} minTickGap={25} stroke="#E2E8F4" />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#8896A8' }} stroke="#E2E8F4" unit="%" />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div style={{ background: 'white', padding: '10px 14px', borderRadius: '8px', border: '1px solid #E2E8F4', boxShadow: 'var(--shadow-md)', fontSize: '0.75rem' }}>
                          <div style={{ fontWeight: 700, color: 'var(--color-text-primary)' }}>{data.time}</div>
                          <div style={{ color: data.decision === 'attack' ? 'var(--color-danger)' : 'var(--color-success)', fontWeight: 600 }}>
                            {data.attackType} ({data.decision?.toUpperCase()})
                          </div>
                          <div>Confidence: {data.confidence}%</div>
                          <div>Stage: {data.stage?.toUpperCase()}</div>
                          {data.severity && <div>Severity: {data.severity}</div>}
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="confidence"
                  stroke="#94A3B8"
                  strokeWidth={1.5}
                  isAnimationActive={false}
                  dot={(props: any) => {
                    const { cx, cy, payload } = props;
                    if (cx == null || cy == null) {
                      return <circle key={props.key || Math.random()} cx={0} cy={0} r={0} fill="none" />;
                    }
                    let dotColor = '#10B981'; // normal
                    if (payload?.decision === 'attack') {
                      if (payload.stage === 'rule') {
                        dotColor = payload.severity === 'High' ? '#DC2626' : '#D97706';
                      } else {
                        dotColor = payload.severity === 'High' ? '#7C3AED' : '#2563EB';
                      }
                    }
                    return (
                      <circle
                        key={props.key || `dot-${payload?.time}-${cx}`}
                        cx={cx}
                        cy={cy}
                        r={payload?.decision === 'attack' ? 4 : 2.5}
                        fill={dotColor}
                        stroke="#fff"
                        strokeWidth={1}
                      />
                    );
                  }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        {/* Attack Breakdown & Stage Split */}
        <article className="xl:col-span-4" style={{ ...CARD_STYLE, padding: '20px 24px' }}>
          <div style={SECTION_LABEL_STYLE}>Attack Breakdown</div>
          <h2 style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>
            Attack Types &amp; Decision Stages
          </h2>
          <p style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)', marginTop: '2px' }}>
            Alert distribution across hybrid decision stages.
          </p>

          {/* Stage split pill */}
          <div style={{ marginTop: '14px', padding: '10px 14px', borderRadius: '8px', background: 'var(--color-bg-base)', border: '1px solid var(--color-border)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', fontWeight: 600, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>
              <span>Decision Stage Split</span>
              <span>{alertsTriggered} Alerts</span>
            </div>
            <div style={{ display: 'flex', gap: '8px', marginTop: '6px' }}>
              <div style={{ flex: Math.max(1, stageSplit.rules), height: '8px', borderRadius: '4px', background: '#EA580C' }} title={`Rules: ${stageSplit.rules}`} />
              <div style={{ flex: Math.max(1, stageSplit.ml), height: '8px', borderRadius: '4px', background: '#2563EB' }} title={`ML: ${stageSplit.ml}`} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '0.72rem' }}>
              <span style={{ color: '#EA580C', fontWeight: 600 }}>Rule Stage: {stageSplit.rules} ({alertsTriggered > 0 ? ((stageSplit.rules / alertsTriggered) * 100).toFixed(0) : 0}%)</span>
              <span style={{ color: '#2563EB', fontWeight: 600 }}>ML Stage: {stageSplit.ml} ({alertsTriggered > 0 ? ((stageSplit.ml / alertsTriggered) * 100).toFixed(0) : 0}%)</span>
            </div>
          </div>

          {/* Counts per attack type */}
          <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div style={{ fontSize: '0.68rem', fontWeight: 600, textTransform: 'uppercase', color: 'var(--color-text-muted)' }}>
              Attack Classification Counts
            </div>
            {Object.entries(attackBreakdown).map(([atkType, count]) => {
              const pct = alertsTriggered > 0 ? (count / alertsTriggered) * 100 : 0;
              const color = ATTACK_TYPE_COLORS[atkType] || '#64748B';
              return (
                <div key={atkType} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', padding: '3px 0' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '8px', height: '8px', borderRadius: '2px', background: color }} />
                    <span style={{ color: 'var(--color-text-secondary)', fontWeight: 500 }}>{atkType}</span>
                  </span>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', color: 'var(--color-text-primary)', fontWeight: 600 }}>
                    {count} ({pct.toFixed(0)}%)
                  </span>
                </div>
              );
            })}
          </div>

          {/* Small severity breakdown */}
          <div style={{ marginTop: '16px', paddingTop: '12px', borderTop: '1px dashed var(--color-border)' }}>
            <div style={{ fontSize: '0.68rem', fontWeight: 600, textTransform: 'uppercase', color: 'var(--color-text-muted)', marginBottom: '6px' }}>
              Severity Breakdown
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              <div style={{ padding: '8px', borderRadius: '6px', background: 'rgba(196,30,58,0.08)', border: '1px solid rgba(196,30,58,0.2)' }}>
                <div style={{ fontSize: '0.65rem', color: 'var(--color-danger)', fontWeight: 700, textTransform: 'uppercase' }}>High</div>
                <div style={{ fontSize: '1.1rem', fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, color: 'var(--color-danger)' }}>
                  {severityBreakdown.high}
                </div>
              </div>
              <div style={{ padding: '8px', borderRadius: '6px', background: 'rgba(217,119,6,0.08)', border: '1px solid rgba(217,119,6,0.2)' }}>
                <div style={{ fontSize: '0.65rem', color: '#D97706', fontWeight: 700, textTransform: 'uppercase' }}>Medium</div>
                <div style={{ fontSize: '1.1rem', fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, color: '#D97706' }}>
                  {severityBreakdown.medium}
                </div>
              </div>
            </div>
          </div>
        </article>

        {/* Threat Feed (Full Detail with SHAP & Rule Reason) */}
        <article className="xl:col-span-12" style={{ ...CARD_STYLE, padding: '20px 24px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={SECTION_LABEL_STYLE}>Threat Feed</div>
              <h2 style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                Session Alerts &amp; Decision Explainability
              </h2>
              <p style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)', marginTop: '2px' }}>
                Each alert reveals arbitration ID, decision stage, severity, rule reason or top-5 SHAP feature contributions.
              </p>
            </div>
            <span style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)' }}>
              Showing last {Math.min(alertsList.length, MAX_FEED_ALERTS)} alerts
            </span>
          </div>

          <div style={{ marginTop: '14px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {alertsList.length === 0 ? (
              <div style={{ border: '1px dashed var(--color-border)', borderRadius: '8px', background: 'var(--color-bg-base)', padding: '20px', textAlign: 'center', fontSize: '0.78rem', color: 'var(--color-text-muted)' }}>
                No threats detected in this session yet. Replay attack classes (DoS, Fuzzy, Spoof_Gear, Spoof_RPM) to view live alert telemetry.
              </div>
            ) : (
              alertsList.slice(-MAX_FEED_ALERTS).reverse().map((alert) => {
                const isExpanded = expandedShapId === alert.id;
                const isRuleStage = alert.stage === 'rule';

                return (
                  <div
                    key={alert.id}
                    style={{
                      border: '1px solid var(--color-border)',
                      borderLeft: alert.severity === 'High' ? '4px solid #DC2626' : '4px solid #D97706',
                      borderRadius: '8px',
                      background: alert.severity === 'High' ? 'rgba(220,38,38,0.03)' : 'rgba(217,119,6,0.03)',
                      padding: '12px 16px',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                        {/* Time */}
                        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                          {alert.time}
                        </span>

                        {/* CAN ID */}
                        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: '0.8rem', color: 'var(--color-text-primary)' }}>
                          {alert.canIdHex} ({alert.canId})
                        </span>

                        {/* Stage Badge */}
                        <span
                          style={{
                            fontSize: '0.62rem',
                            fontWeight: 700,
                            padding: '1px 6px',
                            borderRadius: '4px',
                            letterSpacing: '0.04em',
                            textTransform: 'uppercase',
                            background: isRuleStage ? 'rgba(234,88,12,0.15)' : 'rgba(37,99,235,0.15)',
                            color: isRuleStage ? '#EA580C' : '#2563EB',
                            border: isRuleStage ? '1px solid rgba(234,88,12,0.3)' : '1px solid rgba(37,99,235,0.3)',
                          }}
                        >
                          {alert.stage.toUpperCase()}
                        </span>

                        {/* Attack Type */}
                        <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                          {alert.attackType}
                        </span>

                        {/* Severity Badge */}
                        <span
                          style={{
                            fontSize: '0.62rem',
                            fontWeight: 700,
                            padding: '1px 6px',
                            borderRadius: '4px',
                            letterSpacing: '0.04em',
                            textTransform: 'uppercase',
                            background: alert.severity === 'High' ? 'rgba(220,38,38,0.15)' : 'rgba(217,119,6,0.15)',
                            color: alert.severity === 'High' ? '#DC2626' : '#D97706',
                          }}
                        >
                          {alert.severity ?? 'High'}
                        </span>

                        {/* Simulated Tag */}
                        {alert.simulated && (
                          <span
                            style={{
                              fontSize: '0.6rem',
                              fontWeight: 700,
                              padding: '1px 5px',
                              borderRadius: '4px',
                              background: 'rgba(168,85,247,0.15)',
                              color: '#A855F7',
                              border: '1px solid rgba(168,85,247,0.3)',
                              letterSpacing: '0.04em',
                            }}
                          >
                            SIMULATED
                          </span>
                        )}
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.75rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                          Confidence: {(alert.confidence * 100).toFixed(1)}%
                        </span>

                        {/* Expandable SHAP button for ML alerts */}
                        {!isRuleStage && alert.topFeatures && alert.topFeatures.length > 0 && (
                          <button
                            type="button"
                            onClick={() => setExpandedShapId(isExpanded ? null : alert.id)}
                            style={{
                              padding: '2px 8px',
                              borderRadius: '4px',
                              border: '1px solid var(--color-border)',
                              background: 'var(--color-bg-base)',
                              color: 'var(--color-primary)',
                              fontSize: '0.7rem',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '4px',
                            }}
                          >
                            SHAP (Top 5) {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Rule Stage Reason */}
                    {isRuleStage && alert.reason && (
                      <div style={{ marginTop: '6px', fontSize: '0.72rem', color: '#C2410C', fontFamily: 'JetBrains Mono, monospace' }}>
                        Rule [{alert.ruleId}]: {alert.reason}
                      </div>
                    )}

                    {/* Expandable SHAP Table for ML Alerts */}
                    {!isRuleStage && isExpanded && alert.topFeatures && (
                      <div style={{ marginTop: '10px', paddingTop: '8px', borderTop: '1px dashed var(--color-border)' }}>
                        <div style={{ fontSize: '0.68rem', fontWeight: 600, color: 'var(--color-text-muted)', marginBottom: '4px' }}>
                          Top 5 SHAP Feature Contributions:
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '6px' }}>
                          {alert.topFeatures.slice(0, 5).map((feat, idx) => (
                            <div
                              key={idx}
                              style={{
                                padding: '4px 8px',
                                borderRadius: '4px',
                                background: 'var(--color-bg-base)',
                                border: '1px solid var(--color-border)',
                                fontSize: '0.7rem',
                                fontFamily: 'JetBrains Mono, monospace',
                              }}
                            >
                              <div style={{ color: 'var(--color-text-muted)', fontSize: '0.62rem' }}>{feat.feature}</div>
                              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '2px' }}>
                                <span>Val: {feat.value}</span>
                                <span style={{ color: feat.shap_value > 0 ? '#DC2626' : '#2563EB', fontWeight: 600 }}>
                                  {feat.shap_value > 0 ? `+${feat.shap_value.toFixed(3)}` : feat.shap_value.toFixed(3)}
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </article>
      </section>

      {/* Advanced Telemetry: Byte-Level Delta Map & IAT Distribution */}
      <div style={{ marginTop: '16px' }}>
        <AdvancedAnalytics recentFrames={telemetryFrames} />
      </div>
    </div>
  );
}
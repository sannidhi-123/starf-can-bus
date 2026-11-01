import React, { useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ReferenceLine,
  CartesianGrid,
} from 'recharts';
import {
  HelpCircle,
  Activity,
  ShieldAlert,
  ShieldCheck,
  Split,
  Info,
} from 'lucide-react';
import type { SessionFrame } from '../pages/DashboardPage';

export interface FeatureMeta {
  displayName: string;
  shortName: string;
  description: string;
  category: 'Arbitration' | 'Temporal' | 'Payload';
}

export const CAN_FEATURE_METADATA: Record<string, FeatureMeta> = {
  can_id: {
    displayName: 'CAN Arbitration ID',
    shortName: 'Arbitration ID',
    description:
      'Identifier of transmitting ECU / CAN message function (11-bit standard). Unwhitelisted or unexpected IDs indicate spoofing or DoS message injection.',
    category: 'Arbitration',
  },
  id: {
    displayName: 'CAN Arbitration ID',
    shortName: 'Arbitration ID',
    description:
      'Identifier of transmitting ECU / CAN message function (11-bit standard). Unwhitelisted or unexpected IDs indicate spoofing or DoS message injection.',
    category: 'Arbitration',
  },
  id_repetition_count: {
    displayName: 'ID Burst Repetition',
    shortName: 'Burst Repetition',
    description:
      'Number of occurrences of this CAN ID in the last 50 sequential frames. High repetition signifies aggressive bus flooding characteristic of DoS or injection attacks.',
    category: 'Temporal',
  },
  inter_arrival_time_delta: {
    displayName: 'Inter-Arrival Time (Δt)',
    shortName: 'Time Delta (Δt)',
    description:
      'Elapsed time in seconds between consecutive frames with the same CAN ID. Abnormally small intervals (<0.001s) indicate high-rate automated message injection.',
    category: 'Temporal',
  },
  dlc: {
    displayName: 'Data Length Code (DLC)',
    shortName: 'Payload DLC',
    description:
      'Payload size in bytes (typically 8 for standard vehicle bus communications). Anomalous lengths signal protocol fuzzing.',
    category: 'Payload',
  },
  d0: {
    displayName: 'Payload Byte 0 (d0)',
    shortName: 'Byte 0 (d0)',
    description:
      'First payload data byte. Frequently encodes vehicle transmission gear status (e.g. Park/Drive/Neutral) or operational control flags.',
    category: 'Payload',
  },
  d1: {
    displayName: 'Payload Byte 1 (d1)',
    shortName: 'Byte 1 (d1)',
    description:
      'Second payload data byte. Conveys sensor readings or steering control signals depending on arbitration identifier.',
    category: 'Payload',
  },
  d2: {
    displayName: 'Payload Byte 2 (d2)',
    shortName: 'Byte 2 (d2)',
    description:
      'Third payload data byte. Conveys vehicle speed feedback or powertrain status in intra-vehicular networks.',
    category: 'Payload',
  },
  d3: {
    displayName: 'Payload Byte 3 (d3 - RPM MSB)',
    shortName: 'Byte 3 (RPM MSB)',
    description:
      'Fourth payload data byte. In the Car Hacking dataset, conveys the most significant byte (MSB) of engine RPM telemetry.',
    category: 'Payload',
  },
  d4: {
    displayName: 'Payload Byte 4 (d4 - RPM LSB)',
    shortName: 'Byte 4 (RPM LSB)',
    description:
      'Fifth payload data byte. In the Car Hacking dataset, conveys the least significant byte (LSB) of engine RPM telemetry.',
    category: 'Payload',
  },
  d5: {
    displayName: 'Payload Byte 5 (d5)',
    shortName: 'Byte 5 (d5)',
    description:
      'Sixth payload data byte. Conveys secondary sensor telemetry, pedal position, or auxiliary subsystem status.',
    category: 'Payload',
  },
  d6: {
    displayName: 'Payload Byte 6 (d6)',
    shortName: 'Byte 6 (d6)',
    description:
      'Seventh payload data byte. Conveys CRC checksum or alive counter verification on modern automotive busses.',
    category: 'Payload',
  },
  d7: {
    displayName: 'Payload Byte 7 (d7)',
    shortName: 'Byte 7 (d7)',
    description:
      'Eighth payload data byte. Conveys trailing flags or end-of-frame parity bits.',
    category: 'Payload',
  },
};

export function getFeatureMeta(featureKey: string): FeatureMeta {
  const normalized = featureKey.toLowerCase().trim();
  return (
    CAN_FEATURE_METADATA[normalized] || {
      displayName: featureKey,
      shortName: featureKey,
      description: `Raw telemetry feature: ${featureKey}`,
      category: 'Payload',
    }
  );
}

interface ShapExplanationViewProps {
  alert: SessionFrame;
  baselineNormalFrame?: SessionFrame | null;
}

export const ShapExplanationView: React.FC<ShapExplanationViewProps> = ({
  alert,
  baselineNormalFrame,
}) => {
  const [viewMode, setViewMode] = useState<'attribution' | 'comparison'>('attribution');
  const [activeTooltipFeature, setActiveTooltipFeature] = useState<string | null>(null);

  const isRuleStage = alert.stage === 'rule';
  const hasShap = Boolean(alert.topFeatures && alert.topFeatures.length > 0);
  const targetClass = alert.attackType && alert.attackType !== 'Rule violation' ? alert.attackType : 'Attack';
  const isTargetNormal = targetClass === 'Normal';

  // Format features for horizontal bar chart
  const chartData = (alert.topFeatures || []).slice(0, 5).map((f) => {
    const meta = getFeatureMeta(f.feature);
    return {
      feature: f.feature,
      displayName: meta.shortName,
      fullName: meta.displayName,
      description: meta.description,
      category: meta.category,
      value: f.value,
      shap_value: f.shap_value,
      abs_shap: Math.abs(f.shap_value),
    };
  });

  // Baseline normal chart data (if available)
  const baselineChartData = (baselineNormalFrame?.topFeatures || []).slice(0, 5).map((f) => {
    const meta = getFeatureMeta(f.feature);
    return {
      feature: f.feature,
      displayName: meta.shortName,
      fullName: meta.displayName,
      description: meta.description,
      category: meta.category,
      value: f.value,
      shap_value: f.shap_value,
      abs_shap: Math.abs(f.shap_value),
    };
  });

  return (
    <div
      style={{
        marginTop: '12px',
        paddingTop: '12px',
        borderTop: '1px solid var(--color-border)',
        fontSize: '0.75rem',
      }}
    >
      {/* ── SECTION 1: WHY WAS THIS FRAME FLAGGED? ─────────────────── */}
      <div
        style={{
          borderRadius: '8px',
          border: '1px solid var(--color-border)',
          background: 'var(--color-bg-base)',
          padding: '12px 14px',
          marginBottom: '12px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {isRuleStage ? (
              <ShieldAlert size={16} color="#EA580C" />
            ) : (
              <ShieldCheck size={16} color="#2563EB" />
            )}
            <span style={{ fontWeight: 700, fontSize: '0.78rem', color: 'var(--color-text-primary)' }}>
              Detection Cause: {isRuleStage ? 'Stage 1 Deterministic Rule Engine' : 'Stage 2 Multiclass Random Forest Classifier'}
            </span>
          </div>

          {/* View mode toggle (if baseline available or applicable) */}
          {hasShap && baselineNormalFrame && (
            <div style={{ display: 'flex', gap: '4px', background: 'var(--color-bg-card)', padding: '2px', borderRadius: '6px', border: '1px solid var(--color-border)' }}>
              <button
                type="button"
                onClick={() => setViewMode('attribution')}
                style={{
                  padding: '3px 10px',
                  borderRadius: '4px',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '0.68rem',
                  fontWeight: 600,
                  background: viewMode === 'attribution' ? 'var(--color-primary)' : 'transparent',
                  color: viewMode === 'attribution' ? '#FFFFFF' : 'var(--color-text-muted)',
                }}
              >
                SHAP Attribution
              </button>
              <button
                type="button"
                onClick={() => setViewMode('comparison')}
                style={{
                  padding: '3px 10px',
                  borderRadius: '4px',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '0.68rem',
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  background: viewMode === 'comparison' ? 'var(--color-primary)' : 'transparent',
                  color: viewMode === 'comparison' ? '#FFFFFF' : 'var(--color-text-muted)',
                }}
              >
                <Split size={12} />
                Normal vs Attack
              </button>
            </div>
          )}
        </div>

        {/* Detailed Explanation Text */}
        <div style={{ marginTop: '8px', lineHeight: 1.5, color: 'var(--color-text-secondary)', fontSize: '0.72rem' }}>
          {isRuleStage ? (
            <div>
              <p style={{ margin: 0 }}>
                <strong style={{ color: '#C2410C' }}>Rule Fired: [{alert.ruleId}]</strong> — {alert.reason}
              </p>
              <p style={{ margin: '4px 0 0 0', color: 'var(--color-text-muted)' }}>
                This frame violated deterministic automotive safety rules (0ms latency check).
                {alert.attackType && alert.attackType !== 'Rule violation' && (
                  <span>
                    {' '}Secondary evaluation by the Random Forest classifier categorized this injection pattern as{' '}
                    <strong style={{ color: 'var(--color-text-primary)' }}>{alert.attackType}</strong> (confidence: {(alert.confidence * 100).toFixed(1)}%).
                  </span>
                )}
              </p>
              <div
                style={{
                  marginTop: '6px',
                  padding: '6px 10px',
                  borderRadius: '5px',
                  background: 'rgba(234, 88, 12, 0.08)',
                  border: '1px solid rgba(234, 88, 12, 0.2)',
                  fontSize: '0.68rem',
                  color: '#9A3412',
                }}
              >
                <Info size={12} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'text-bottom' }} />
                <strong>Interpretability Note:</strong> SHAP values explain feature contributions for the{' '}
                <strong>Random Forest ML classifier</strong> output. The alert itself was triggered independently by the deterministic rule engine.
              </div>
            </div>
          ) : (
            <div>
              <p style={{ margin: 0 }}>
                Frame successfully passed all Stage 1 deterministic rule tests (valid whitelisted ID, legitimate frequency, and valid inter-arrival timing).
              </p>
              <p style={{ margin: '4px 0 0 0' }}>
                <strong style={{ color: '#2563EB' }}>ML Classification:</strong> Random Forest identified subtle payload / temporal anomaly signatures matching{' '}
                <strong style={{ color: 'var(--color-text-primary)' }}>{alert.attackType}</strong> with{' '}
                <strong>{(alert.confidence * 100).toFixed(1)}%</strong> confidence.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ── SECTION 2: SHAP VISUAL ATTRIBUTION CHART ──────────────── */}
      {hasShap ? (
        viewMode === 'attribution' ? (
          <div
            style={{
              borderRadius: '8px',
              border: '1px solid var(--color-border)',
              background: 'var(--color-bg-card)',
              padding: '14px 16px',
            }}
          >
            {/* Header & Directionality Legend */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px', marginBottom: '12px' }}>
              <div>
                <div style={{ fontWeight: 700, fontSize: '0.78rem', color: 'var(--color-text-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Activity size={14} color="var(--color-primary)" />
                  TreeSHAP Local Feature Attributions (Top 5)
                </div>
                <div style={{ fontSize: '0.68rem', color: 'var(--color-text-muted)', marginTop: '2px' }}>
                  Explaining Random Forest classification relative to class: <strong style={{ color: 'var(--color-text-primary)' }}>{targetClass}</strong>
                </div>
              </div>

              {/* Legend */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '0.68rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: isTargetNormal ? '#10B981' : '#EF4444' }} />
                  <span style={{ color: 'var(--color-text-secondary)' }}>
                    Positive (+): Pushes <strong>toward {targetClass}</strong>
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#3B82F6' }} />
                  <span style={{ color: 'var(--color-text-secondary)' }}>
                    Negative (-): Pushes <strong>away from {targetClass}</strong>
                  </span>
                </div>
              </div>
            </div>

            {/* Horizontal Bar Chart (Recharts) */}
            <div style={{ width: '100%', height: 210 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  layout="vertical"
                  data={chartData}
                  margin={{ top: 10, right: 35, left: 25, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--color-border)" />
                  <XAxis
                    type="number"
                    domain={['auto', 'auto']}
                    tick={{ fontSize: 10, fill: 'var(--color-text-muted)' }}
                    tickFormatter={(val) => (val >= 0 ? `+${val.toFixed(2)}` : val.toFixed(2))}
                  />
                  <YAxis
                    type="category"
                    dataKey="displayName"
                    tick={{ fontSize: 10, fill: 'var(--color-text-primary)', fontWeight: 600 }}
                    width={110}
                  />
                  <ReferenceLine x={0} stroke="var(--color-text-muted)" strokeWidth={1.5} />
                  <Tooltip
                    content={({ active, payload }) => {
                      if (!active || !payload || !payload.length) return null;
                      const item = payload[0].payload;
                      const isPos = item.shap_value >= 0;
                      return (
                        <div
                          style={{
                            background: 'var(--color-bg-card)',
                            border: '1px solid var(--color-border)',
                            borderRadius: '6px',
                            padding: '8px 12px',
                            boxShadow: 'var(--shadow-md)',
                            fontSize: '0.72rem',
                            maxWidth: '280px',
                          }}
                        >
                          <div style={{ fontWeight: 700, color: 'var(--color-text-primary)' }}>
                            {item.fullName}
                          </div>
                          <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.65rem', color: 'var(--color-text-muted)' }}>
                            Key: {item.feature} | Value: {item.value}
                          </div>
                          <div style={{ margin: '6px 0', borderTop: '1px dashed var(--color-border)' }} />
                          <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
                            <span>TreeSHAP Attribution (φ):</span>
                            <strong style={{ color: isPos ? (isTargetNormal ? '#10B981' : '#EF4444') : '#3B82F6' }}>
                              {isPos ? `+${item.shap_value.toFixed(4)}` : item.shap_value.toFixed(4)}
                            </strong>
                          </div>
                          <div style={{ marginTop: '4px', fontSize: '0.65rem', color: 'var(--color-text-muted)' }}>
                            {isPos
                              ? `Supports model decision to classify this frame as ${targetClass}.`
                              : `Counter-evidence pushing model away from ${targetClass}.`}
                          </div>
                          <div style={{ marginTop: '6px', fontSize: '0.65rem', color: 'var(--color-text-secondary)', background: 'var(--color-bg-base)', padding: '4px 6px', borderRadius: '4px' }}>
                            {item.description}
                          </div>
                        </div>
                      );
                    }}
                  />
                  <Bar dataKey="shap_value" radius={[3, 3, 3, 3]}>
                    {chartData.map((entry, index) => {
                      const isPos = entry.shap_value >= 0;
                      const fill = isPos ? (isTargetNormal ? '#10B981' : '#EF4444') : '#3B82F6';
                      return <Cell key={`cell-${index}`} fill={fill} />;
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Feature Attribution Details Table */}
            <div style={{ marginTop: '14px' }}>
              <div style={{ fontSize: '0.68rem', fontWeight: 600, color: 'var(--color-text-muted)', marginBottom: '6px' }}>
                Feature Values &amp; Contribution Breakdown:
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '6px' }}>
                {chartData.map((feat, idx) => {
                  const isPos = feat.shap_value >= 0;
                  const isTooltipOpen = activeTooltipFeature === feat.feature;

                  return (
                    <div
                      key={idx}
                      style={{
                        padding: '6px 10px',
                        borderRadius: '6px',
                        background: 'var(--color-bg-base)',
                        border: '1px solid var(--color-border)',
                        position: 'relative',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontWeight: 600, color: 'var(--color-text-primary)', fontSize: '0.72rem' }}>
                          {feat.displayName}
                        </span>
                        <button
                          type="button"
                          onClick={() => setActiveTooltipFeature(isTooltipOpen ? null : feat.feature)}
                          style={{
                            background: 'transparent',
                            border: 'none',
                            cursor: 'pointer',
                            color: 'var(--color-text-muted)',
                            padding: '1px',
                            display: 'flex',
                          }}
                          title={feat.description}
                        >
                          <HelpCircle size={12} />
                        </button>
                      </div>

                      <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.65rem', color: 'var(--color-text-muted)', marginTop: '2px' }}>
                        {feat.feature} = <span style={{ color: 'var(--color-text-primary)', fontWeight: 600 }}>{feat.value}</span>
                      </div>

                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '4px' }}>
                        <span style={{ fontSize: '0.65rem', color: 'var(--color-text-muted)' }}>Contribution:</span>
                        <span
                          style={{
                            fontFamily: 'JetBrains Mono, monospace',
                            fontWeight: 700,
                            fontSize: '0.72rem',
                            color: isPos ? (isTargetNormal ? '#10B981' : '#EF4444') : '#3B82F6',
                          }}
                        >
                          {isPos ? `+${feat.shap_value.toFixed(4)}` : feat.shap_value.toFixed(4)}
                        </span>
                      </div>

                      {/* Expanded description popup if clicked */}
                      {isTooltipOpen && (
                        <div
                          style={{
                            marginTop: '6px',
                            padding: '6px 8px',
                            borderRadius: '4px',
                            background: 'var(--color-bg-card)',
                            border: '1px solid var(--color-border)',
                            fontSize: '0.65rem',
                            color: 'var(--color-text-secondary)',
                            lineHeight: 1.4,
                          }}
                        >
                          {feat.description}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        ) : (
          /* ── SECTION 3: NORMAL VS ATTACK COMPARISON VIEW ──────────── */
          <div
            style={{
              borderRadius: '8px',
              border: '1px solid var(--color-border)',
              background: 'var(--color-bg-card)',
              padding: '14px 16px',
            }}
          >
            <div style={{ marginBottom: '12px' }}>
              <div style={{ fontWeight: 700, fontSize: '0.78rem', color: 'var(--color-text-primary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Split size={14} color="var(--color-primary)" />
                Side-by-Side Comparison: Normal Baseline vs Detected Attack
              </div>
              <div style={{ fontSize: '0.68rem', color: 'var(--color-text-muted)', marginTop: '2px' }}>
                Contrasting feature values and calculated TreeSHAP contributions from authentic Car Hacking test data.
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
              {/* Left Column: Normal Frame */}
              <div style={{ border: '1px solid var(--color-border)', borderRadius: '6px', padding: '10px 12px', background: 'var(--color-bg-base)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--color-border)', paddingBottom: '6px', marginBottom: '8px' }}>
                  <div>
                    <span style={{ fontWeight: 700, color: '#10B981', fontSize: '0.75rem' }}>Baseline Normal Frame</span>
                    <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.68rem', color: 'var(--color-text-muted)' }}>
                      ID: {baselineNormalFrame?.canIdHex} ({baselineNormalFrame?.canId})
                    </div>
                  </div>
                  <span style={{ fontSize: '0.65rem', padding: '2px 6px', borderRadius: '4px', background: 'rgba(16,185,129,0.1)', color: '#10B981', fontWeight: 700 }}>
                    NORMAL ({( (baselineNormalFrame?.confidence ?? 0.8) * 100 ).toFixed(1)}%)
                  </span>
                </div>

                <div style={{ width: '100%', height: 160 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart layout="vertical" data={baselineChartData} margin={{ top: 5, right: 25, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--color-border)" />
                      <XAxis type="number" tick={{ fontSize: 9, fill: 'var(--color-text-muted)' }} />
                      <YAxis type="category" dataKey="displayName" tick={{ fontSize: 9, fill: 'var(--color-text-primary)' }} width={85} />
                      <ReferenceLine x={0} stroke="var(--color-text-muted)" />
                      <Bar dataKey="shap_value" radius={[2, 2, 2, 2]}>
                        {baselineChartData.map((e, idx) => (
                          <Cell key={idx} fill={e.shap_value >= 0 ? '#10B981' : '#3B82F6'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p style={{ fontSize: '0.65rem', color: 'var(--color-text-muted)', margin: '4px 0 0 0' }}>
                  Regular timing intervals, expected burst frequency, and standard operating telemetry drive positive attribution toward Normal.
                </p>
              </div>

              {/* Right Column: Attack Frame */}
              <div style={{ border: '1px solid var(--color-border)', borderRadius: '6px', padding: '10px 12px', background: 'var(--color-bg-base)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--color-border)', paddingBottom: '6px', marginBottom: '8px' }}>
                  <div>
                    <span style={{ fontWeight: 700, color: '#EF4444', fontSize: '0.75rem' }}>Detected Attack Frame</span>
                    <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.68rem', color: 'var(--color-text-muted)' }}>
                      ID: {alert.canIdHex} ({alert.canId})
                    </div>
                  </div>
                  <span style={{ fontSize: '0.65rem', padding: '2px 6px', borderRadius: '4px', background: 'rgba(239,68,68,0.1)', color: '#EF4444', fontWeight: 700 }}>
                    {alert.attackType.toUpperCase()} ({(alert.confidence * 100).toFixed(1)}%)
                  </span>
                </div>

                <div style={{ width: '100%', height: 160 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart layout="vertical" data={chartData} margin={{ top: 5, right: 25, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--color-border)" />
                      <XAxis type="number" tick={{ fontSize: 9, fill: 'var(--color-text-muted)' }} />
                      <YAxis type="category" dataKey="displayName" tick={{ fontSize: 9, fill: 'var(--color-text-primary)' }} width={85} />
                      <ReferenceLine x={0} stroke="var(--color-text-muted)" />
                      <Bar dataKey="shap_value" radius={[2, 2, 2, 2]}>
                        {chartData.map((e, idx) => (
                          <Cell key={idx} fill={e.shap_value >= 0 ? '#EF4444' : '#3B82F6'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p style={{ fontSize: '0.65rem', color: 'var(--color-text-muted)', margin: '4px 0 0 0' }}>
                  Abnormal repetition bursts, compressed inter-arrival Δt, and manipulated byte payloads drive decisive attribution toward {alert.attackType}.
                </p>
              </div>
            </div>

            {/* Contrast Summary Banner */}
            <div
              style={{
                marginTop: '10px',
                padding: '8px 12px',
                borderRadius: '6px',
                background: 'var(--color-bg-base)',
                border: '1px solid var(--color-border)',
                fontSize: '0.68rem',
                color: 'var(--color-text-secondary)',
              }}
            >
              <strong style={{ color: 'var(--color-primary)' }}>Key Contrast Insight:</strong> In legitimate driving, arbitration IDs repeat at regulated intervals (~10-50ms) with low window repetition counts. Under attack (such as {alert.attackType}), messages flood the vehicle bus with near-zero Δt or injected spoof payloads, causing TreeSHAP to assign dominant positive attributions to temporal repetition and specific modified data bytes.
            </div>
          </div>
        )
      ) : (
        <div
          style={{
            padding: '12px',
            borderRadius: '6px',
            background: 'var(--color-bg-base)',
            border: '1px dashed var(--color-border)',
            color: 'var(--color-text-muted)',
            fontSize: '0.72rem',
            textAlign: 'center',
          }}
        >
          {isRuleStage ? (
            <span>
              SHAP feature attributions are unavailable for this frame because it was flagged directly by Stage 1 deterministic rules without an ML attack classification.
            </span>
          ) : (
            <span>SHAP feature attribution explainer is currently offline or loading.</span>
          )}
        </div>
      )}
    </div>
  );
};

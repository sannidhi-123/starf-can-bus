import { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

export type FrameTelemetry = {
  timestamp: number;
  canId: string;
  canIdHex: string;
  dataBytes: number[];
};

export type AdvancedAnalyticsProps = {
  recentFrames?: FrameTelemetry[];
};

function getHeatColor(value: number): string {
  const normalized = Math.min(1, Math.max(0, value / 255));
  if (normalized < 0.1) return '#dbeaff';
  if (normalized < 0.3) return '#9fc4f4';
  if (normalized < 0.5) return '#4E89D6';
  if (normalized < 0.75) return '#2a73cc';
  return '#1f5fa8';
}

const IAT_LABELS = ['< 1ms', '1-2ms', '2-5ms', '5-10ms', '10-20ms', '20-50ms', '50-100ms', '100ms+'] as const;

export function AdvancedAnalytics({ recentFrames = [] }: AdvancedAnalyticsProps) {
  // Use last up to 64 frames received in this session
  const activeWindow = useMemo(() => {
    return recentFrames.slice(-64);
  }, [recentFrames]);

  // Compute real absolute byte deltas capped to 0..255
  const byteDeltas = useMemo(() => {
    if (activeWindow.length === 0) {
      return [];
    }

    const deltas: Array<{ row: number; col: number; value: number }> = [];
    for (let r = 0; r < activeWindow.length; r++) {
      const curr = activeWindow[r].dataBytes || [0, 0, 0, 0, 0, 0, 0, 0];
      const prev = r > 0 ? activeWindow[r - 1].dataBytes || [0, 0, 0, 0, 0, 0, 0, 0] : curr;
      for (let c = 0; c < 8; c++) {
        const diff = Math.min(255, Math.abs((curr[c] ?? 0) - (prev[c] ?? 0)));
        deltas.push({
          row: r,
          col: c,
          value: diff,
        });
      }
    }
    return deltas;
  }, [activeWindow]);

  // Compute real IAT histogram from consecutive frames received in the session
  const iatData = useMemo(() => {
    if (recentFrames.length < 2) {
      return [];
    }

    const counts: Record<string, number> = {
      '< 1ms': 0,
      '1-2ms': 0,
      '2-5ms': 0,
      '5-10ms': 0,
      '10-20ms': 0,
      '20-50ms': 0,
      '50-100ms': 0,
      '100ms+': 0,
    };

    for (let i = 1; i < recentFrames.length; i++) {
      const deltaSec = Math.abs(recentFrames[i].timestamp - recentFrames[i - 1].timestamp);
      const deltaMs = deltaSec * 1000;

      if (deltaMs < 1) counts['< 1ms']++;
      else if (deltaMs < 2) counts['1-2ms']++;
      else if (deltaMs < 5) counts['2-5ms']++;
      else if (deltaMs < 10) counts['5-10ms']++;
      else if (deltaMs < 20) counts['10-20ms']++;
      else if (deltaMs < 50) counts['20-50ms']++;
      else if (deltaMs < 100) counts['50-100ms']++;
      else counts['100ms+']++;
    }

    return IAT_LABELS.map((label) => ({
      bin: label,
      count: counts[label] || 0,
    }));
  }, [recentFrames]);

  const maxIAT = useMemo(() => {
    if (iatData.length === 0) return 1;
    return Math.max(1, ...iatData.map((d) => d.count));
  }, [iatData]);

  return (
    <section className="mt-8">
      <div className="mb-6 flex items-center gap-3">
        <span className="material-symbols-outlined text-secondary">insights</span>
        <h2 className="font-headline text-xl font-bold text-primary">Advanced Network Analysis Tools</h2>
      </div>
      <p className="mb-8 max-w-3xl text-sm text-on-surface-variant">
        Supplementary telemetry instruments built from the frames received in this session. Inspect real payload byte deltas
        and real inter-frame arrival time intervals across live traffic.
      </p>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Chart A: Byte-Level Delta Map */}
        <div className="rounded-xl bg-surface-container-lowest p-6 border border-outline-variant/30">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="font-headline text-sm font-bold text-primary">Byte-Level Delta Map</h3>
            <span className="rounded bg-secondary-fixed px-2 py-1 font-mono text-[10px] font-bold uppercase tracking-widest text-secondary">
              {activeWindow.length > 0 ? `Last ${activeWindow.length} Frames` : 'Window (up to 64)'}
            </span>
          </div>
          <p className="mb-4 text-xs text-on-surface-variant">
            Heatmap displaying real absolute byte-value deltas (|B<sub>t</sub> - B<sub>t-1</sub>|) capped at 0–255 across frames received in this session.
          </p>

          {activeWindow.length === 0 ? (
            <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-outline-variant/40 bg-surface-container-low text-xs text-on-surface-variant">
              No frames yet
            </div>
          ) : (
            <div className="overflow-hidden rounded-lg border border-outline-variant/20">
              <div className="bg-surface-container-low px-3 py-2">
                <div className="grid grid-cols-8 gap-0 text-center font-mono text-[9px] text-on-surface-variant">
                  {Array.from({ length: 8 }, (_, i) => (
                    <div key={`col-header-${i}`}>B{i}</div>
                  ))}
                </div>
              </div>
              <div
                className="grid grid-cols-8 gap-[1px] bg-surface-container-low p-[1px] max-h-64 overflow-y-auto"
                style={{ gridAutoRows: 'minmax(18px, auto)' }}
              >
                {byteDeltas.map((cell) => (
                  <div
                    key={`cell-${cell.row}-${cell.col}`}
                    className="flex aspect-square items-center justify-center font-mono text-[9px] font-bold text-white transition-opacity hover:opacity-80"
                    style={{ backgroundColor: getHeatColor(cell.value) }}
                    title={`Frame ${cell.row}, Byte ${cell.col}: Δ${cell.value}`}
                  >
                    {cell.value}
                  </div>
                ))}
              </div>
              <div className="flex items-center justify-between bg-surface-container-low px-3 py-2">
                <span className="font-mono text-[9px] text-on-surface-variant">Low Δ (0)</span>
                <div className="flex gap-0.5">
                  {['#dbeaff', '#9fc4f4', '#4E89D6', '#2a73cc', '#1f5fa8'].map((color) => (
                    <div key={color} className="h-2 w-6 rounded-sm" style={{ backgroundColor: color }} />
                  ))}
                </div>
                <span className="font-mono text-[9px] text-on-surface-variant">High Δ (255)</span>
              </div>
            </div>
          )}

          <div className="mt-4 rounded-lg bg-surface-container-low p-3">
            <p className="font-mono text-[10px] text-on-surface-variant">
              Statistical baseline (from models/rules_config.json): Normal CAN 2.0B traffic exhibits stable payload signatures restricted to 31 legitimate CAN IDs. Rapid uniform shifts (0xFF) or repetitive bit mutations indicate injection attacks.
            </p>
          </div>
        </div>

        {/* Chart B: Inter-Frame Arrival Time Histogram */}
        <div className="rounded-xl bg-surface-container-lowest p-6 border border-outline-variant/30">
          <div className="mb-2 flex items-center justify-between">
            <h3 className="font-headline text-sm font-bold text-primary">Inter-Frame Arrival Time (IAT) Distribution</h3>
            <span className="rounded bg-secondary-fixed px-2 py-1 font-mono text-[10px] font-bold uppercase tracking-widest text-secondary">
              {recentFrames.length > 1 ? `${recentFrames.length} Frames` : 'Session Histogram'}
            </span>
          </div>
          <p className="mb-4 text-xs text-on-surface-variant">
            Empirical distribution of inter-frame arrival times measured between consecutive frames received during this session.
          </p>

          {recentFrames.length < 2 ? (
            <div className="flex h-56 items-center justify-center rounded-lg border border-dashed border-outline-variant/40 bg-surface-container-low text-xs text-on-surface-variant">
              No frames yet
            </div>
          ) : (
            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={iatData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#d7dee8" vertical={false} />
                  <XAxis
                    dataKey="bin"
                    tick={{ fontSize: 10, fill: '#5f6773' }}
                    stroke="#d7dee8"
                    axisLine={{ stroke: '#d7dee8' }}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: '#5f6773' }}
                    stroke="#d7dee8"
                    axisLine={{ stroke: '#d7dee8' }}
                    allowDecimals={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #d7dee8',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                    formatter={(value: number) => [`${value.toLocaleString()} frames`, 'Count']}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {iatData.map((entry, index) => {
                      const ratio = entry.count / maxIAT;
                      const color = ratio > 0.7 ? '#2a73cc' : ratio > 0.3 ? '#4E89D6' : '#9fc4f4';
                      return <Cell key={`iat-bar-${index}`} fill={color} />;
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="mt-4 rounded-lg bg-surface-container-low p-3">
            <p className="font-mono text-[10px] text-on-surface-variant">
              Statistical baseline (from models/rules_config.json): Normal CAN 2.0B traffic has an empirical default minimum IAT of 3.76ms (timing threshold 0.1 percentile with 0.7 margin). Sub-millisecond clustering characterizes high-frequency DoS flooding.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

// FILE: frontend/src/pages/AnalyzePage.tsx - BUGS FIXED: 4, 6
import { FormEvent, useMemo, useState } from 'react';
import { Play, Radar } from 'lucide-react';
import { apiClient, type PredictionResponse } from '../api/client';
import { useLiveStreamContext } from '../context/LiveStreamContext';
import { AppLayout } from '../layout/AppLayout';

function parseSequenceInput(text: string): string[] {
  return text
    .split(/[\n,\s]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

const CARD_STYLE: React.CSSProperties = {
  background: 'var(--color-bg-card)',
  border: '1px solid var(--color-border)',
  borderRadius: '10px',
  boxShadow: 'var(--shadow-card)',
  padding: '20px 24px',
};

const SECTION_LABEL_STYLE: React.CSSProperties = {
  fontSize: '0.68rem',
  fontWeight: 600,
  textTransform: 'uppercase',
  letterSpacing: '0.09em',
  color: 'var(--color-text-muted)',
  marginBottom: '12px',
  paddingLeft: '8px',
  borderLeft: '3px solid var(--color-primary)',
};

export function AnalyzePage() {
  const [sequenceText, setSequenceText] = useState('');
  const [analysisMode, setAnalysisMode] = useState<'fast' | 'detailed'>('fast');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const { points } = useLiveStreamContext();
  const tokens = useMemo(() => parseSequenceInput(sequenceText), [sequenceText]);

  const feedRows = useMemo(() => {
    const latest = [...points].slice(-6).reverse();
    return latest.map((entry) => ({
      id: entry.id,
      time: entry.time,
      score: entry.score,
      label: entry.label,
      type: entry.attackType,
    }));
  }, [points]);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();

    if (tokens.length === 0) {
      setError('At least one CAN ID is required.');
      return;
    }

    setError(null);
    setLoading(true);

    try {
      const prediction = await apiClient.predict(tokens, analysisMode === 'detailed');
      setResult(prediction);
    } catch (submitError) {
      const message = submitError instanceof Error ? submitError.message : 'Inference pipeline execution failed.';
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  const resultBoxStyle: React.CSSProperties = result?.is_attack
    ? {
        background: 'var(--color-danger-bg)',
        border: '1px solid rgba(196,30,58,0.2)',
        borderLeft: '4px solid var(--color-danger)',
        borderRadius: '10px',
        padding: '16px 20px',
      }
    : {
        background: 'var(--color-success-bg)',
        border: '1px solid rgba(15,123,85,0.2)',
        borderLeft: '4px solid var(--color-success)',
        borderRadius: '10px',
        padding: '16px 20px',
      };

  return (
    <AppLayout>
      <div style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box', minHeight: 0 }}>
        <div style={{ marginBottom: '20px' }}>
          <h1
            style={{
              fontSize: '1.1rem',
              fontWeight: 700,
              color: 'var(--color-text-primary)',
              letterSpacing: '-0.01em',
              lineHeight: 1.2,
            }}
          >
            CAN Frame &amp; Sequence Analyzer
          </h1>
          <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '3px' }}>
            Submit CAN arbitration IDs for hybrid rule heuristics and multiclass ML evaluation
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
          <section className="xl:col-span-8" style={CARD_STYLE}>
            <div style={SECTION_LABEL_STYLE}>Manual Input</div>
            <h2 style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>Hybrid inference</h2>
            <p style={{ marginTop: '4px', fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
              Enter single or multiple CAN IDs (hex or decimal) to evaluate through the STARF hybrid detection pipeline.
            </p>

            <form onSubmit={onSubmit} style={{ marginTop: '14px' }}>
              <label
                htmlFor="sequence-input"
                style={{
                  display: 'block',
                  marginBottom: '8px',
                  fontSize: '0.68rem',
                  fontWeight: 500,
                  textTransform: 'uppercase',
                  letterSpacing: '0.09em',
                  color: 'var(--color-text-muted)',
                }}
              >
                CAN Arbitration IDs
              </label>

              <textarea
                id="sequence-input"
                className="analyze-sequence-input"
                rows={7}
                placeholder="0x18F 0x0C9 0x111 0x000 ..."
                value={sequenceText}
                onChange={(event) => setSequenceText(event.target.value)}
              />

              <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px', flexWrap: 'wrap' }}>
                <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>{tokens.length} ID{tokens.length === 1 ? '' : 's'} entered</span>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <button
                    type="button"
                    onClick={() => setAnalysisMode('fast')}
                    style={{
                      border: '1px solid var(--color-border)',
                      background: analysisMode === 'fast' ? 'var(--color-primary-light)' : 'var(--color-bg-card)',
                      color: analysisMode === 'fast' ? 'var(--color-primary)' : 'var(--color-text-secondary)',
                      borderRadius: '8px',
                      padding: '8px 12px',
                      fontSize: '0.75rem',
                      fontWeight: 500,
                      cursor: 'pointer',
                    }}
                  >
                    Fast
                  </button>
                  <button
                    type="button"
                    onClick={() => setAnalysisMode('detailed')}
                    style={{
                      border: '1px solid var(--color-border)',
                      background: analysisMode === 'detailed' ? 'var(--color-primary-light)' : 'var(--color-bg-card)',
                      color: analysisMode === 'detailed' ? 'var(--color-primary)' : 'var(--color-text-secondary)',
                      borderRadius: '8px',
                      padding: '8px 12px',
                      fontSize: '0.75rem',
                      fontWeight: 500,
                      cursor: 'pointer',
                    }}
                  >
                    Detailed
                  </button>
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="mt-3 inline-flex items-center gap-2 rounded-btn bg-primary px-6 py-2.5 text-[0.83rem] font-semibold text-white transition-colors hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-70"
              >
                <Play size={14} />
                {loading ? 'Analyzing...' : 'Analyze Input'}
              </button>

              {error ? <p style={{ marginTop: '10px', fontSize: '0.76rem', color: 'var(--color-danger)' }}>{error}</p> : null}
            </form>
          </section>

          <section className="xl:col-span-4" style={CARD_STYLE}>
            <div style={SECTION_LABEL_STYLE}>Result</div>
            <h2 style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>Inference output</h2>
            <p style={{ marginTop: '4px', fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>Current prediction snapshot.</p>

            <div style={{ marginTop: '14px', ...resultBoxStyle }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', gap: '8px' }}>
                <span
                  style={{
                    fontSize: '0.68rem',
                    fontWeight: 600,
                    textTransform: 'uppercase',
                    letterSpacing: '0.09em',
                    color: result?.is_attack ? 'var(--color-danger)' : 'var(--color-success)',
                  }}
                >
                  {result ? (result.is_attack ? 'Attack Detected' : 'Normal Traffic') : 'Awaiting Input'}
                </span>
                <Radar size={16} color={result?.is_attack ? 'var(--color-danger)' : 'var(--color-success)'} />
              </div>

              <div style={{ marginTop: '8px', fontFamily: 'JetBrains Mono, monospace', fontSize: '1.3rem', fontWeight: 700, color: 'var(--color-text-primary)' }}>
                {result ? `${((result.confidence ?? 0) * 100).toFixed(1)}%` : '0.0%'}
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)' }}>Classification Confidence</div>

              <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                <div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--color-text-muted)' }}>Stage</div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.8rem', color: 'var(--color-text-primary)', fontWeight: 600 }}>
                    {result ? (result.stage?.toUpperCase() || (result.rule_id ? 'RULE' : 'ML')) : 'N/A'}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--color-text-muted)' }}>Attack Type</div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.8rem', color: 'var(--color-text-primary)', fontWeight: 600 }}>{result?.attack_type ?? 'N/A'}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--color-text-muted)' }}>Severity</div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.8rem', color: 'var(--color-text-primary)' }}>
                    {result?.severity ?? (result?.is_attack ? 'High' : 'Normal')}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.68rem', color: 'var(--color-text-muted)' }}>Latency</div>
                  <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.8rem', color: 'var(--color-text-primary)' }}>
                    {result ? `${result.processing_time_ms.toFixed(1)}ms` : '0.0ms'}
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="xl:col-span-12" style={CARD_STYLE}>
            <div style={SECTION_LABEL_STYLE}>Live Feed</div>
            <h2 style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>Recent telemetry rows</h2>
            <p style={{ marginTop: '4px', fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>Most recent streaming decisions from dashboard context.</p>

            <div style={{ marginTop: '12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {feedRows.length === 0 ? (
                <div style={{ border: '1px solid var(--color-border)', borderRadius: '8px', background: 'var(--color-bg-base)', padding: '10px 14px', fontSize: '0.76rem', color: 'var(--color-text-secondary)' }}>
                  Waiting for live stream data...
                </div>
              ) : (
                feedRows.map((row) => (
                  <div
                    key={row.id}
                    style={{
                      border: '1px solid var(--color-border)',
                      borderRadius: '8px',
                      background: 'var(--color-bg-base)',
                      padding: '10px 14px',
                      display: 'flex',
                      justifyContent: 'space-between',
                      gap: '10px',
                      fontSize: '0.76rem',
                      color: 'var(--color-text-secondary)',
                    }}
                  >
                    <span style={{ minWidth: 0 }}>
                      {row.label === 'ATTACK' ? `${row.type ?? 'Unknown'} attack detected` : 'Normal traffic'}
                    </span>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.72rem', color: 'var(--color-text-primary)' }}>
                      {row.score.toFixed(3)}
                    </span>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      </div>
    </AppLayout>
  );
}
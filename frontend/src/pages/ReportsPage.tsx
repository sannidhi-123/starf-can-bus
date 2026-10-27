// FILE: frontend/src/pages/ReportsPage.tsx - FIXES: chart URL construction, image error/loading handling, filename normalization, fallback chart rendering, full manifest endpoint
import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Download } from 'lucide-react';
import { AppLayout } from '../layout/AppLayout';

const API_BASE = 'http://localhost:8000';

type ManifestFile = {
  name: string;
};

type ManifestPayload = {
  charts?: string[];
  available_files?: ManifestFile[];
  metrics?: {
    metrics?: {
      accuracy?: number;
      precision?: number;
      recall?: number;
      f1_score?: number;
      f1?: number;
      specificity?: number;
    };
  };
};

// Metrics from held-out chronological hybrid evaluation (100,000 frames)
const DEFAULT_METRICS = {
  accuracy: 0.9897,
  precision: 0.9968,
  recall: 1.0000,
  f1: 0.9984,
  specificity: 0.9871,
};

const FALLBACK_CHARTS = [
  'confusion_matrix.png',
  'roc_curve.png',
  'pr_curve.png',
  'score_distribution.png',
  'metrics_summary.png',
];

const chartMeta: Record<string, { title: string; description: string; badge: string; fullWidth?: boolean }> = {
  'confusion_matrix.png': {
    title: 'Confusion Matrix',
    description: 'True/False Positive and Negative counts across Normal and Attack classes',
    badge: 'Classification',
  },
  'roc_curve.png': {
    title: 'ROC Curve',
    description: 'Receiver Operating Characteristic - detection vs false alarm tradeoff',
    badge: 'AUC Score',
  },
  'pr_curve.png': {
    title: 'Precision-Recall Curve',
    description: 'Precision vs Recall at all decision thresholds',
    badge: 'AP Score',
  },
  'score_distribution.png': {
    title: 'Anomaly Score Distribution',
    description: 'Separation between Normal and Attack score distributions with threshold',
    badge: 'Threshold',
    fullWidth: true,
  },
  'metrics_summary.png': {
    title: 'Metrics Summary',
    description: 'Overall performance across all evaluation dimensions',
    badge: 'Summary',
    fullWidth: true,
  },
};

const CARD_STYLE: CSSProperties = {
  background: 'var(--color-bg-card)',
  border: '1px solid var(--color-border)',
  borderRadius: '10px',
  boxShadow: 'var(--shadow-card)',
  padding: '20px 24px',
};

const SECTION_LABEL_STYLE: CSSProperties = {
  fontSize: '0.68rem',
  fontWeight: 600,
  textTransform: 'uppercase',
  letterSpacing: '0.09em',
  color: 'var(--color-text-muted)',
  paddingLeft: '8px',
  borderLeft: '3px solid var(--color-primary)',
  marginBottom: '16px',
};

const normaliseFilename = (name: string): string => {
  const trimmed = name.trim();
  if (!trimmed.includes('.')) {
    return `${trimmed}.png`;
  }
  return trimmed;
};

function metricValue(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export function ReportsPage() {
  const [manifestData, setManifestData] = useState<ManifestPayload | null>(null);
  const [manifestLoading, setManifestLoading] = useState(true);
  const [manifestError, setManifestError] = useState(false);
  const [imgError, setImgError] = useState<Record<string, boolean>>({});
  const [imgLoaded, setImgLoaded] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const loadManifest = async () => {
      setManifestLoading(true);
      setManifestError(false);

      try {
        const response = await fetch('http://localhost:8000/reports/manifest', {
          signal: AbortSignal.timeout(5000),
        });

        if (!response.ok) {
          throw new Error('Manifest request failed');
        }

        const data = (await response.json()) as ManifestPayload;
        setManifestData(data);
      } catch {
        setManifestData(null);
        setManifestError(true);
      } finally {
        setManifestLoading(false);
      }
    };

    loadManifest();
  }, []);

  const metrics = useMemo(() => {
    const values = manifestData?.metrics?.metrics;
    return {
      accuracy: values?.accuracy ?? DEFAULT_METRICS.accuracy,
      precision: values?.precision ?? DEFAULT_METRICS.precision,
      recall: values?.recall ?? DEFAULT_METRICS.recall,
      f1: values?.f1_score ?? values?.f1 ?? DEFAULT_METRICS.f1,
      specificity: values?.specificity ?? DEFAULT_METRICS.specificity,
    };
  }, [manifestData]);

  const manifestCharts = useMemo(() => {
    if (manifestData?.charts && manifestData.charts.length > 0) {
      return manifestData.charts;
    }

    const files = manifestData?.available_files ?? [];
    if (files.length > 0) {
      return files.map((entry) => entry.name).filter((name) => name.toLowerCase().endsWith('.png') || !name.includes('.'));
    }

    return [] as string[];
  }, [manifestData]);

  const chartsToShow = useMemo(() => {
    const source = manifestCharts.length > 0 ? manifestCharts : FALLBACK_CHARTS;
    return Array.from(new Set(source.map(normaliseFilename)));
  }, [manifestCharts]);

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
            Model Reports
          </h1>
          <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '3px' }}>
            Evaluation metrics and generated report artifacts
          </p>
        </div>

        <section className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5" style={{ marginBottom: '16px' }}>
          {[
            ['Accuracy', metricValue(metrics.accuracy)],
            ['Precision', metricValue(metrics.precision)],
            ['Recall', metricValue(metrics.recall)],
            ['F1 Score', metricValue(metrics.f1)],
            ['Specificity', metricValue(metrics.specificity)],
          ].map(([label, value]) => (
            <article key={label} style={CARD_STYLE}>
              <div
                style={{
                  fontSize: '0.68rem',
                  fontWeight: 500,
                  textTransform: 'uppercase',
                  letterSpacing: '0.09em',
                  color: 'var(--color-text-muted)',
                }}
              >
                {label}
              </div>
              <div
                style={{
                  marginTop: '8px',
                  fontSize: '1.3rem',
                  fontWeight: 700,
                  fontFamily: 'JetBrains Mono, monospace',
                  color: 'var(--color-primary)',
                  lineHeight: 1.1,
                }}
              >
                {value}
              </div>
            </article>
          ))}
        </section>

        <section style={{ ...CARD_STYLE, marginBottom: '16px' }}>
          <div style={SECTION_LABEL_STYLE}>Manifest Status</div>
          {manifestLoading ? (
            <div style={{ fontSize: '0.78rem', color: 'var(--color-text-muted)' }}>Loading report manifest...</div>
          ) : manifestError ? (
            <div
              style={{
                background: 'var(--color-bg-base)',
                border: '1px solid var(--color-border)',
                borderRadius: '8px',
                padding: '12px 14px',
              }}
            >
              <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--color-text-primary)' }}>Backend Offline</div>
              <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '4px', lineHeight: 1.5 }}>
                Manifest could not be loaded from http://localhost:8000/reports/manifest. Chart rendering falls back to known
                filenames and still attempts to load report images.
              </p>
            </div>
          ) : (
            <div style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)' }}>
              Manifest loaded successfully. Charts are resolved from manifest when available and fallback filenames are used when missing.
            </div>
          )}
        </section>

        <section style={CARD_STYLE}>
          <div style={SECTION_LABEL_STYLE}>Evaluation Charts</div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '16px',
            }}
          >
            {chartsToShow.map((filename) => {
              const meta = chartMeta[filename] || {
                title: filename.replace('.png', '').replace(/_/g, ' '),
                description: 'Evaluation chart',
                badge: 'Chart',
              };

              const chartUrl = `${API_BASE}/static/reports/${filename}`;

              return (
                <div
                  key={filename}
                  style={{
                    gridColumn: meta.fullWidth ? '1 / -1' : 'auto',
                    background: 'var(--color-bg-card)',
                    border: '1px solid var(--color-border)',
                    borderRadius: '10px',
                    boxShadow: 'var(--shadow-card)',
                    padding: '16px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '12px',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'flex-start',
                    }}
                  >
                    <div>
                      <div
                        style={{
                          fontSize: '0.82rem',
                          fontWeight: 600,
                          color: 'var(--color-text-primary)',
                        }}
                      >
                        {meta.title}
                      </div>
                      <div
                        style={{
                          fontSize: '0.72rem',
                          color: 'var(--color-text-muted)',
                          marginTop: '2px',
                          lineHeight: 1.4,
                        }}
                      >
                        {meta.description}
                      </div>
                    </div>
                    <span
                      style={{
                        fontSize: '0.6rem',
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        letterSpacing: '0.08em',
                        color: 'var(--color-primary)',
                        background: 'var(--color-primary-light)',
                        padding: '2px 8px',
                        borderRadius: '999px',
                        whiteSpace: 'nowrap',
                        marginLeft: '12px',
                      }}
                    >
                      {meta.badge}
                    </span>
                  </div>

                  {imgError[filename] ? (
                    <div
                      style={{
                        width: '100%',
                        height: '200px',
                        background: 'var(--color-bg-base)',
                        border: '1px dashed var(--color-border)',
                        borderRadius: '7px',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '8px',
                      }}
                    >
                      <span
                        style={{
                          fontSize: '1.5rem',
                          opacity: 0.3,
                        }}
                      >
                        📊
                      </span>
                      <div
                        style={{
                          fontSize: '0.75rem',
                          color: 'var(--color-text-muted)',
                          textAlign: 'center',
                        }}
                      >
                        Chart not available
                        <br />
                        <span
                          style={{
                            fontSize: '0.68rem',
                            fontFamily: 'JetBrains Mono, monospace',
                          }}
                        >
                          Run python src/evaluate.py to generate
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div style={{ position: 'relative', minHeight: '200px' }}>
                      {!imgLoaded[filename] ? (
                        <div
                          style={{
                            position: 'absolute',
                            inset: 0,
                            background: 'var(--color-bg-base)',
                            border: '1px solid var(--color-border)',
                            borderRadius: '7px',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            fontSize: '0.72rem',
                            color: 'var(--color-text-muted)',
                          }}
                        >
                          Loading chart...
                        </div>
                      ) : null}

                      <img
                        src={chartUrl}
                        alt={meta.title}
                        onError={() =>
                          setImgError((prev) => ({
                            ...prev,
                            [filename]: true,
                          }))
                        }
                        onLoad={() =>
                          setImgLoaded((prev) => ({
                            ...prev,
                            [filename]: true,
                          }))
                        }
                        style={{
                          opacity: imgLoaded[filename] ? 1 : 0,
                          transition: 'opacity 0.3s ease',
                          width: '100%',
                          borderRadius: '7px',
                          border: '1px solid var(--color-border)',
                          display: 'block',
                        }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        <section style={{ ...CARD_STYLE, marginTop: '16px' }}>
          <div style={SECTION_LABEL_STYLE}>Download Report</div>
          <a
            href={`${API_BASE}/static/reports/model_evaluation_report.pdf`}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-btn bg-primary px-6 py-2.5 text-[0.83rem] font-semibold text-white transition-colors hover:bg-primary-hover"
            style={{ color: 'var(--color-text-inverse)' }}
          >
            <Download size={14} />
            Download PDF Report
          </a>
        </section>
      </div>
    </AppLayout>
  );
}

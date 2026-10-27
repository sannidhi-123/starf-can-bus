// =============================================================================
// ThreatLog.tsx — Terminal-style Threat Detection Log
// Shows: decision, stage, rule reason, attack type, severity badge,
//        confidence, top 5 SHAP features, and SIMULATED tag.
// =============================================================================

import React, { useRef, useEffect, useState } from 'react';
import {
  AlertTriangle,
  Terminal,
  Trash2,
  Clock,
  Hash,
  Gauge,
  Shield,
  ChevronDown,
  ChevronUp,
  Activity,
  Zap,
  Eye,
} from 'lucide-react';
import type { ThreatLogEntry, Severity, ShapFeature } from '../types';

interface ThreatLogProps {
  entries: ThreatLogEntry[];
  onClear: () => void;
}

// ---------------------------------------------------------------------------
// Severity badge
// ---------------------------------------------------------------------------
function SeverityBadge({ severity }: { severity: Severity }) {
  if (!severity) return null;

  const styles: Record<string, string> = {
    High:   'bg-red-600/20 text-red-400 border border-red-600/40',
    Medium: 'bg-orange-500/20 text-orange-400 border border-orange-500/40',
  };

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold uppercase tracking-wide ${styles[severity] ?? ''}`}
    >
      {severity}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Stage badge
// ---------------------------------------------------------------------------
function StageBadge({ stage }: { stage: 'rule' | 'ml' | 'legacy' }) {
  const map: Record<string, { label: string; cls: string }> = {
    rule:   { label: 'RULE',   cls: 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/40' },
    ml:     { label: 'ML',     cls: 'bg-blue-500/20 text-blue-300 border border-blue-500/40' },
    legacy: { label: 'LEGACY', cls: 'bg-gray-500/20 text-gray-400 border border-gray-500/40' },
  };
  const { label, cls } = map[stage] ?? map['legacy'];
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold ${cls}`}>
      {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// SHAP feature mini-bar list
// ---------------------------------------------------------------------------
function ShapBar({ features }: { features: ShapFeature[] }) {
  if (!features || features.length === 0) return null;

  const maxAbs = Math.max(...features.map(f => Math.abs(f.shap_value)), 1e-9);

  return (
    <div className="mt-2 space-y-1">
      <p className="text-xs text-cyber-muted font-semibold uppercase tracking-wide mb-1 flex items-center gap-1">
        <Activity className="w-3 h-3" />
        Top SHAP Features
      </p>
      {features.map((f) => {
        const pct = Math.abs(f.shap_value) / maxAbs;
        const positive = f.shap_value >= 0;
        return (
          <div key={f.feature} className="flex items-center gap-2 text-xs">
            <span className="text-cyber-muted w-36 truncate shrink-0">{f.feature}</span>
            <div className="flex-1 bg-cyber-surface rounded-full h-1.5 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-300 ${positive ? 'bg-red-500' : 'bg-blue-400'}`}
                style={{ width: `${pct * 100}%` }}
              />
            </div>
            <span className={`w-16 text-right tabular-nums ${positive ? 'text-red-400' : 'text-blue-400'}`}>
              {f.shap_value >= 0 ? '+' : ''}{f.shap_value.toFixed(4)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Single log entry card (expandable)
// ---------------------------------------------------------------------------
function LogEntry({ entry, index }: { entry: ThreatLogEntry; index: number }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className="log-entry log-entry-threat rounded-lg animate-slide-in"
      style={{ animationDelay: `${index * 50}ms` }}
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 flex-wrap">
          <AlertTriangle className="w-4 h-4 text-threat-critical animate-pulse shrink-0" />
          <span className="text-threat-critical font-semibold">INTRUSION DETECTED</span>
          <StageBadge stage={entry.stage} />
          {entry.severity && <SeverityBadge severity={entry.severity} />}
          {entry.simulated && (
            <span className="px-2 py-0.5 rounded-full text-xs font-bold bg-purple-600/20 text-purple-400 border border-purple-600/40 uppercase tracking-wide">
              SIMULATED
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-cyber-muted text-xs shrink-0">
          <Clock className="w-3 h-3" />
          <span>{entry.localTime}</span>
          <button
            onClick={() => setExpanded(v => !v)}
            className="p-0.5 rounded hover:text-white transition-colors"
            aria-label="Toggle details"
          >
            {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
        </div>
      </div>

      {/* Primary metrics */}
      <div className="grid grid-cols-3 gap-4 text-xs">
        <div className="flex items-center gap-2">
          <Hash className="w-3 h-3 text-cyber-muted" />
          <span className="text-cyber-muted">CAN ID:</span>
          <span className="text-white font-bold">{entry.canId}</span>
        </div>
        <div className="flex items-center gap-2">
          <Gauge className="w-3 h-3 text-cyber-muted" />
          <span className="text-cyber-muted">Score:</span>
          <span className="text-threat-critical font-bold">
            {entry.anomalyScore.toFixed(4)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Shield className="w-3 h-3 text-cyber-muted" />
          <span className="text-cyber-muted">Conf:</span>
          <span className="text-white">{(entry.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Attack type row */}
      {entry.attackType && (
        <div className="mt-1 flex items-center gap-2 text-xs">
          <Zap className="w-3 h-3 text-orange-400 shrink-0" />
          <span className="text-cyber-muted">Type:</span>
          <span className="text-orange-300 font-semibold">{entry.attackType}</span>
        </div>
      )}

      {/* Rule reason row (shown inline when rule stage) */}
      {entry.stage === 'rule' && entry.reason && (
        <div className="mt-1 flex items-start gap-2 text-xs">
          <Eye className="w-3 h-3 text-yellow-400 mt-0.5 shrink-0" />
          <span className="text-cyber-muted shrink-0">Rule:</span>
          <span className="text-yellow-300 leading-snug">{entry.reason}</span>
        </div>
      )}

      {/* Expanded details */}
      {expanded && (
        <div className="mt-2 pt-2 border-t border-threat-critical/20 space-y-2">
          {/* Stage & rule_id */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <span className="text-cyber-muted">Stage: </span>
              <span className="text-white">{entry.stage}</span>
            </div>
            {entry.ruleId && (
              <div>
                <span className="text-cyber-muted">Rule ID: </span>
                <span className="text-yellow-300 font-mono">{entry.ruleId}</span>
              </div>
            )}
          </div>

          {/* Full reason for rule path */}
          {entry.stage === 'rule' && entry.reason && (
            <div className="text-xs">
              <span className="text-cyber-muted">Reason: </span>
              <span className="text-yellow-200">{entry.reason}</span>
            </div>
          )}

          {/* SHAP features for ML path */}
          {entry.stage === 'ml' && entry.topFeatures.length > 0 && (
            <ShapBar features={entry.topFeatures} />
          )}

          {/* Timestamp */}
          <div className="text-xs text-cyber-muted opacity-70">{entry.timestamp}</div>
        </div>
      )}

      {/* Collapsed footer */}
      {!expanded && (
        <div className="mt-2 pt-2 border-t border-threat-critical/20 text-xs text-cyber-muted">
          <span className="opacity-70">{entry.timestamp}</span>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ThreatLog component
// ---------------------------------------------------------------------------
export const ThreatLog: React.FC<ThreatLogProps> = ({ entries, onClear }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [entries]);

  return (
    <div className="glass-card h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-cyber-border">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-threat-critical/10 flex items-center justify-center">
            <Terminal className="w-5 h-5 text-threat-critical" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white flex items-center gap-2">
              Threat Log
              {entries.length > 0 && (
                <span className="px-2 py-0.5 rounded-full bg-threat-critical/20 text-threat-critical text-xs font-mono">
                  {entries.length}
                </span>
              )}
            </h3>
            <p className="text-sm text-cyber-muted">Security events · click entry for details</p>
          </div>
        </div>

        <button
          onClick={onClear}
          disabled={entries.length === 0}
          className={`
            flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium
            transition-all duration-200
            ${entries.length > 0
              ? 'bg-cyber-surface hover:bg-cyber-border text-white'
              : 'bg-cyber-surface/50 text-cyber-muted cursor-not-allowed'
            }
          `}
        >
          <Trash2 className="w-4 h-4" />
          Clear
        </button>
      </div>

      {/* Log entries */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-4 space-y-2 font-mono text-sm"
        style={{ maxHeight: '400px' }}
      >
        {entries.length === 0 ? (
          <div className="h-full flex items-center justify-center text-cyber-muted">
            <div className="text-center">
              <p>No threats detected</p>
              <p className="text-xs mt-1">System is operating normally</p>
            </div>
          </div>
        ) : (
          entries.map((entry, index) => (
            <LogEntry key={entry.id} entry={entry} index={index} />
          ))
        )}
      </div>

      {/* Footer stats */}
      {entries.length > 0 && (
        <div className="p-3 border-t border-cyber-border bg-cyber-darker/50">
          <div className="flex items-center justify-between text-xs text-cyber-muted">
            <span>Latest: {entries[0]?.localTime ?? 'N/A'}</span>
            <span>Total: {entries.length}</span>
            <span>
              Avg score:{' '}
              {(entries.reduce((s, e) => s + e.anomalyScore, 0) / entries.length).toFixed(2)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ThreatLog;

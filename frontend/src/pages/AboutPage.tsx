// FILE: frontend/src/pages/AboutPage.tsx - FIXES: structured left-aligned layout, inline SVG diagrams, section hierarchy, expanded security and pipeline documentation
import { type CSSProperties, type ReactNode } from 'react';
import { Activity, Database, Info, Layers, ShieldAlert } from 'lucide-react';
import { AppLayout } from '../layout/AppLayout';

const PAGE_CONTAINER_STYLE: CSSProperties = {
  maxWidth: '860px',
  margin: '0 auto',
  width: '100%',
};

const SECTION_CARD_STYLE: CSSProperties = {
  background: 'white',
  border: '1px solid var(--color-border)',
  borderRadius: '12px',
  boxShadow: 'var(--shadow-card)',
  padding: '24px 28px',
  marginBottom: '20px',
};

const SECTION_HEADER_ROW_STYLE: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '10px',
  marginBottom: '16px',
  paddingBottom: '12px',
  borderBottom: '1px solid var(--color-border)',
};

const SECTION_ICON_WRAP_STYLE: CSSProperties = {
  width: '32px',
  height: '32px',
  background: 'var(--color-primary-light)',
  borderRadius: '8px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  flexShrink: 0,
};

const SECTION_TITLE_STYLE: CSSProperties = {
  fontSize: '0.95rem',
  fontWeight: 700,
  color: 'var(--color-text-primary)',
  letterSpacing: '-0.01em',
};

const SECTION_SUBTITLE_STYLE: CSSProperties = {
  fontSize: '0.72rem',
  color: 'var(--color-text-muted)',
  marginTop: '1px',
};

const BODY_TEXT_STYLE: CSSProperties = {
  fontSize: '0.83rem',
  lineHeight: 1.7,
  color: 'var(--color-text-secondary)',
  textAlign: 'left',
};

function SectionHeader({ icon, title, subtitle }: { icon: ReactNode; title: string; subtitle: string }) {
  return (
    <div style={SECTION_HEADER_ROW_STYLE}>
      <div style={SECTION_ICON_WRAP_STYLE}>{icon}</div>
      <div>
        <div style={SECTION_TITLE_STYLE}>{title}</div>
        <div style={SECTION_SUBTITLE_STYLE}>{subtitle}</div>
      </div>
    </div>
  );
}

export function AboutPage() {
  return (
    <AppLayout>
      <div style={PAGE_CONTAINER_STYLE}>
        <div style={{ marginBottom: '20px' }}>
          <h1
            style={{
              fontSize: '1.1rem',
              fontWeight: 700,
              color: 'var(--color-text-primary)',
              letterSpacing: '-0.01em',
              lineHeight: 1.2,
              margin: 0,
            }}
          >
            About This System
          </h1>
          <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '3px' }}>
            Architecture overview, detection methodology, and technical documentation
          </p>
        </div>

        <section style={SECTION_CARD_STYLE}>
          <SectionHeader
            icon={<Info size={16} color="var(--color-primary)" />}
            title="Project Overview"
            subtitle="What this system does and why it exists"
          />

          <div
            style={{
              background: 'var(--color-primary-light)',
              borderLeft: '4px solid var(--color-primary)',
              borderRadius: '0 8px 8px 0',
              padding: '14px 18px',
              marginBottom: '16px',
            }}
          >
            <p
              style={{
                fontSize: '0.88rem',
                fontWeight: 500,
                color: 'var(--color-primary)',
                lineHeight: 1.5,
                fontStyle: 'italic',
                margin: 0,
              }}
            >
              "If a CAN bus speaks a language, a hacker speaks broken grammar. We teach a Transformer to learn that language - and
              catch every typo."
            </p>
          </div>

          <p style={{ ...BODY_TEXT_STYLE, margin: 0 }}>
            Modern vehicles contain between 70 and 100 Electronic Control Units (ECUs) that communicate thousands of times per second
            over the Controller Area Network (CAN) bus. This 1986-era protocol was designed without authentication, encryption, or
            message integrity verification - making it inherently vulnerable to cyber attacks. A single compromised OBD-II port or
            Bluetooth module gives an attacker unrestricted write access to safety-critical systems including brakes, steering, and
            throttle.
          </p>
          <p style={{ ...BODY_TEXT_STYLE, marginTop: '12px', marginBottom: 0 }}>
            STARF (Smart Traffic Anomaly Recognition Kit) implements a high-performance Hybrid Intrusion Detection System.
            Stage 1 uses deterministic automotive rule heuristics (arbitration ID whitelist validation, frequency monitors,
            and inter-arrival timing checks) fitted on training Normal traffic. Unflagged traffic passes to Stage 2, which uses
            a multiclass Random Forest ML classifier trained on frame-level repetition, inter-arrival time deltas, and payload bytes
            to classify cyber-attacks (DoS, Fuzzy, Spoof_Gear, Spoof_RPM). The legacy sequence-based DistilBERT MLM path is retained
            strictly as an offline research baseline.
          </p>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: '12px',
              marginTop: '20px',
            }}
          >
            {[
              { value: '56M+', label: 'Training Sequences', color: 'var(--color-primary)' },
              { value: '95%+', label: 'F1 Detection Score', color: 'var(--color-success)' },
              { value: '~1%', label: 'False Alarm Rate', color: 'var(--color-warning)' },
            ].map((stat) => (
              <div
                key={stat.label}
                style={{
                  background: 'var(--color-bg-base)',
                  border: '1px solid var(--color-border)',
                  borderRadius: '8px',
                  padding: '14px 16px',
                  textAlign: 'center',
                }}
              >
                <div
                  style={{
                    fontSize: '1.4rem',
                    fontWeight: 700,
                    fontFamily: 'JetBrains Mono, monospace',
                    color: stat.color,
                    lineHeight: 1.1,
                  }}
                >
                  {stat.value}
                </div>
                <div
                  style={{
                    marginTop: '6px',
                    fontSize: '0.68rem',
                    textTransform: 'uppercase',
                    letterSpacing: '0.08em',
                    color: 'var(--color-text-muted)',
                  }}
                >
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section style={SECTION_CARD_STYLE}>
          <SectionHeader
            icon={<Layers size={16} color="var(--color-primary)" />}
            title="System Architecture"
            subtitle="Two-stage detection pipeline overview"
          />

          <svg
            viewBox="0 0 820 120"
            xmlns="http://www.w3.org/2000/svg"
            style={{ width: '100%', height: 'auto', marginBottom: '20px' }}
          >
            <rect width="820" height="120" fill="#F4F6FB" rx="10" />

            <rect x="10" y="30" width="90" height="60" rx="7" fill="#EBF4FF" stroke="#1A6FD4" strokeWidth="1.5" />
            <text x="55" y="55" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#1A6FD4">
              Raw CAN
            </text>
            <text x="55" y="68" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#1A6FD4">
              CSV Data
            </text>
            <text x="55" y="83" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#8896A8">
              18.3M rows
            </text>

            <line x1="100" y1="60" x2="118" y2="60" stroke="#1A6FD4" strokeWidth="1.5" markerEnd="url(#arrow)" />

            <rect x="120" y="30" width="90" height="60" rx="7" fill="#EBF4FF" stroke="#1A6FD4" strokeWidth="1.5" />
            <text x="165" y="55" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#1A6FD4">
              Preprocess
            </text>
            <text x="165" y="68" textAnchor="middle" fontSize="8" fontFamily="Inter,sans-serif" fill="#4A5568">
              Clean+Tokenize
            </text>
            <text x="165" y="83" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#8896A8">
              vocab.json
            </text>

            <line x1="210" y1="60" x2="228" y2="60" stroke="#1A6FD4" strokeWidth="1.5" markerEnd="url(#arrow)" />

            <rect x="230" y="30" width="90" height="60" rx="7" fill="#EBF4FF" stroke="#1A6FD4" strokeWidth="1.5" />
            <text x="275" y="55" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#1A6FD4">
              Sliding
            </text>
            <text x="275" y="68" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#1A6FD4">
              Window
            </text>
            <text x="275" y="83" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#8896A8">
              size=64
            </text>

            <line x1="320" y1="60" x2="338" y2="60" stroke="#1A6FD4" strokeWidth="1.5" markerEnd="url(#arrow)" />

            <rect x="340" y="20" width="100" height="80" rx="7" fill="#1A6FD4" stroke="#1560BC" strokeWidth="1.5" />
            <text x="390" y="47" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="700" fill="white">
              Hybrid Rules
            </text>
            <text x="390" y="60" textAnchor="middle" fontSize="8" fontFamily="Inter,sans-serif" fill="#EBF4FF">
              Stage 1 Rules
            </text>
            <text x="390" y="73" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#EBF4FF">
              rules_config.json
            </text>
            <text x="390" y="86" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#EBF4FF">
              31 Whitelist IDs
            </text>

            <line x1="440" y1="60" x2="458" y2="60" stroke="#1A6FD4" strokeWidth="1.5" markerEnd="url(#arrow)" />

            <rect x="460" y="30" width="90" height="60" rx="7" fill="#FFF0F0" stroke="#C41E3A" strokeWidth="1.5" />
            <text x="505" y="52" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#C41E3A">
              Threshold
            </text>
            <text x="505" y="65" textAnchor="middle" fontSize="8" fontFamily="Inter,sans-serif" fill="#4A5568">
              99th pct
            </text>
            <text x="505" y="80" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#8896A8">
              score {'>'} tau?
            </text>

            <line x1="505" y1="90" x2="505" y2="108" stroke="#0F7B55" strokeWidth="1.5" markerEnd="url(#arrowGreen)" />
            <text x="490" y="107" fontSize="8" fontFamily="Inter,sans-serif" fill="#0F7B55" fontWeight="600">
              NORMAL
            </text>

            <line x1="550" y1="60" x2="568" y2="60" stroke="#C41E3A" strokeWidth="1.5" markerEnd="url(#arrowRed)" />
            <text x="555" y="52" fontSize="8" fontFamily="Inter,sans-serif" fill="#C41E3A" fontWeight="600">
              ATTACK
            </text>

            <rect x="570" y="30" width="100" height="60" rx="7" fill="#FFF0F0" stroke="#C41E3A" strokeWidth="1.5" />
            <text x="620" y="52" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#C41E3A">
              RandomForest
            </text>
            <text x="620" y="65" textAnchor="middle" fontSize="8" fontFamily="Inter,sans-serif" fill="#4A5568">
              Stage 2 Classify
            </text>
            <text x="620" y="78" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#8896A8">
              TF-IDF + stats
            </text>

            <line x1="670" y1="60" x2="688" y2="60" stroke="#C41E3A" strokeWidth="1.5" markerEnd="url(#arrowRed)" />

            <rect x="690" y="20" width="120" height="80" rx="7" fill="#FFF0F0" stroke="#C41E3A" strokeWidth="1.5" />
            <text x="750" y="44" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="700" fill="#C41E3A">
              Attack Type
            </text>
            <text x="750" y="58" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#4A5568">
              DoS - Fuzzy
            </text>
            <text x="750" y="70" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#4A5568">
              RPM - Gear
            </text>
            <text x="750" y="85" textAnchor="middle" fontSize="8" fontFamily="Inter,sans-serif" fill="#8896A8">
              + confidence
            </text>

            <defs>
              <marker id="arrow" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
                <path d="M0,0 L0,6 L6,3 z" fill="#1A6FD4" />
              </marker>
              <marker id="arrowGreen" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
                <path d="M0,0 L0,6 L6,3 z" fill="#0F7B55" />
              </marker>
              <marker id="arrowRed" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
                <path d="M0,0 L0,6 L6,3 z" fill="#C41E3A" />
              </marker>
            </defs>
          </svg>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '12px',
              marginTop: '4px',
            }}
          >
            <div
              style={{
                background: 'var(--color-bg-base)',
                border: '1px solid var(--color-border)',
                borderRadius: '8px',
                padding: '14px 16px',
              }}
            >
              <div
                style={{
                  fontSize: '0.68rem',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                  color: 'var(--color-primary)',
                  borderLeft: '3px solid var(--color-primary)',
                  paddingLeft: '8px',
                  marginBottom: '12px',
                }}
              >
                Stage 1 - Anomaly Detection
              </div>

              {[
                ['Stage 1 Pipeline', 'Deterministic Rules Engine'],
                ['Heuristics', 'INVALID_ID, FREQUENCY, TIMING'],
                ['Whitelist', '31 legitimate CAN IDs'],
                ['Stage 2 Classifier', 'Random Forest (100 estimators)'],
                ['SMOTE Balancing', 'Applied on train split only'],
                ['Features', '12 frame features (rep, iat, dlc, d0-d7)'],
                ['Held-Out F1 Score', '99.84% (100% recall)'],
              ].map(([key, value]) => (
                <div
                  key={key}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: '0.78rem',
                    padding: '5px 0',
                    borderBottom: '1px dashed var(--color-border)',
                  }}
                >
                  <span style={{ color: 'var(--color-text-muted)', fontWeight: 500 }}>{key}</span>
                  <span style={{ color: 'var(--color-text-primary)', fontFamily: 'JetBrains Mono, monospace' }}>{value}</span>
                </div>
              ))}
            </div>

            <div
              style={{
                background: 'var(--color-bg-base)',
                border: '1px solid var(--color-border)',
                borderRadius: '8px',
                padding: '14px 16px',
              }}
            >
              <div
                style={{
                  fontSize: '0.68rem',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                  color: 'var(--color-danger)',
                  borderLeft: '3px solid var(--color-danger)',
                  paddingLeft: '8px',
                  marginBottom: '12px',
                }}
              >
                Stage 2 - Attack Classification
              </div>

              {[
                ['Classifier', 'RandomForest'],
                ['Features', 'TF-IDF + 6 numeric'],
                ['n-grams', '1-gram + 2-gram'],
                ['Max features', '1,500'],
                ['Classes', 'DoS, Fuzzy, RPM, Gear'],
                ['Trigger', 'Stage 1 anomaly only'],
              ].map(([key, value]) => (
                <div
                  key={key}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: '0.78rem',
                    padding: '5px 0',
                    borderBottom: '1px dashed var(--color-border)',
                  }}
                >
                  <span style={{ color: 'var(--color-text-muted)', fontWeight: 500 }}>{key}</span>
                  <span style={{ color: 'var(--color-text-primary)', fontFamily: 'JetBrains Mono, monospace' }}>{value}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section style={SECTION_CARD_STYLE}>
          <SectionHeader
            icon={<Activity size={16} color="var(--color-primary)" />}
            title="How Detection Works"
            subtitle="The three-step detection process explained"
          />

          <svg viewBox="0 0 700 200" xmlns="http://www.w3.org/2000/svg" style={{ width: '100%', height: 'auto', marginBottom: '16px' }}>
            <rect width="700" height="200" fill="#F4F6FB" rx="10" />

            <circle cx="100" cy="100" r="36" fill="#1A6FD4" stroke="none" />
            <text x="100" y="94" textAnchor="middle" fontSize="20" fontFamily="Inter,sans-serif" fontWeight="800" fill="white">
              1
            </text>
            <text x="100" y="110" textAnchor="middle" fontSize="8" fontFamily="Inter,sans-serif" fontWeight="600" fill="white">
              TOKENISE
            </text>

            <text x="100" y="152" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#0F1729">
              CAN IDs to Tokens
            </text>
            <text x="100" y="165" textAnchor="middle" fontSize="8" fontFamily="Inter,sans-serif" fill="#8896A8">
              hex to integer
            </text>
            <text x="100" y="178" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#8896A8">
              0316 to 3
            </text>

            <line x1="140" y1="100" x2="210" y2="100" stroke="#1A6FD4" strokeWidth="2" strokeDasharray="4,3" markerEnd="url(#arrowBlue2)" />

            <circle cx="250" cy="100" r="36" fill="#1A6FD4" stroke="none" />
            <text x="250" y="94" textAnchor="middle" fontSize="20" fontFamily="Inter,sans-serif" fontWeight="800" fill="white">
              2
            </text>
            <text x="250" y="110" textAnchor="middle" fontSize="8" fontFamily="Inter,sans-serif" fontWeight="600" fill="white">
              WINDOW
            </text>

            <text x="250" y="152" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#0F1729">
              64-Token Sequence
            </text>
            <text x="250" y="165" textAnchor="middle" fontSize="8" fontFamily="Inter,sans-serif" fill="#8896A8">
              sliding window
            </text>
            <text x="250" y="178" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#8896A8">
              step=32
            </text>

            <line x1="290" y1="100" x2="360" y2="100" stroke="#1A6FD4" strokeWidth="2" strokeDasharray="4,3" markerEnd="url(#arrowBlue2)" />

            <circle cx="400" cy="100" r="36" fill="#1A6FD4" stroke="none" />
            <text x="400" y="94" textAnchor="middle" fontSize="20" fontFamily="Inter,sans-serif" fontWeight="800" fill="white">
              3
            </text>
            <text x="400" y="110" textAnchor="middle" fontSize="8" fontFamily="Inter,sans-serif" fontWeight="600" fill="white">
              SCORE
            </text>

            <text x="400" y="152" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#0F1729">
              MLM Reconstruction
            </text>
            <text x="400" y="165" textAnchor="middle" fontSize="8" fontFamily="Inter,sans-serif" fill="#8896A8">
              cross-entropy loss
            </text>
            <text x="400" y="178" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#8896A8">
              to anomaly score
            </text>

            <line x1="440" y1="100" x2="510" y2="100" stroke="#C41E3A" strokeWidth="2" strokeDasharray="4,3" markerEnd="url(#arrowRed2)" />

            <circle cx="550" cy="100" r="36" fill="#C41E3A" stroke="none" />
            <text x="550" y="94" textAnchor="middle" fontSize="20" fontFamily="Inter,sans-serif" fontWeight="800" fill="white">
              4
            </text>
            <text x="550" y="110" textAnchor="middle" fontSize="8" fontFamily="Inter,sans-serif" fontWeight="600" fill="white">
              DECIDE
            </text>

            <text x="550" y="152" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#0F1729">
              score vs threshold
            </text>
            <text x="550" y="165" textAnchor="middle" fontSize="8" fontFamily="Inter,sans-serif" fill="#8896A8">
              99th percentile
            </text>
            <text x="550" y="178" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#8896A8">
              NORMAL / ATTACK
            </text>

            <line x1="590" y1="100" x2="645" y2="100" stroke="#C41E3A" strokeWidth="2" strokeDasharray="4,3" markerEnd="url(#arrowRed2)" />

            <rect x="650" y="64" width="44" height="72" rx="7" fill="#FFF0F0" stroke="#C41E3A" strokeWidth="1.5" />
            <text x="672" y="88" textAnchor="middle" fontSize="16">
              🚨
            </text>
            <text x="672" y="104" textAnchor="middle" fontSize="7" fontFamily="Inter,sans-serif" fontWeight="700" fill="#C41E3A">
              ALERT
            </text>
            <text x="672" y="116" textAnchor="middle" fontSize="7" fontFamily="Inter,sans-serif" fill="#8896A8">
              Stage 2
            </text>
            <text x="672" y="128" textAnchor="middle" fontSize="7" fontFamily="Inter,sans-serif" fill="#8896A8">
              classifies
            </text>

            <defs>
              <marker id="arrowBlue2" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                <path d="M0,0 L0,6 L6,3 z" fill="#1A6FD4" />
              </marker>
              <marker id="arrowRed2" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                <path d="M0,0 L0,6 L6,3 z" fill="#C41E3A" />
              </marker>
            </defs>
          </svg>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: '12px',
            }}
          >
            {[
              {
                title: 'Stage 1: Deterministic Rules Engine',
                borderColor: 'var(--color-primary)',
                text:
                  'Automotive CAN networks operate with strict periodicity and fixed arbitration IDs. Stage 1 enforces a 31-CAN ID whitelist, a 0.5s frequency anomaly window, and empirical inter-arrival time minimums (default 3.76ms) fitted strictly on training Normal traffic. Obvious floods and invalid IDs are flagged instantly with zero model inference latency.',
              },
              {
                title: 'Stage 2: Multiclass ML Classification',
                borderColor: 'var(--color-danger)',
                text:
                  'Frames not flagged by Stage 1 rules proceed to Stage 2. A Random Forest classifier evaluates 12 frame-level features: CAN ID, 50-frame repetition count, inter-arrival time delta, DLC, and payload bytes d0..d7. It accurately distinguishes benign traffic from subtle Fuzzy injections, Gear spoofing, and RPM manipulation.',
              },
              {
                title: 'Decision Architecture & Legacy Fallback',
                borderColor: 'var(--color-warning)',
                text:
                  'Rule hits produce immediate Rule violation alerts with the responsible rule_id, evidence, and severity (High > Medium). Frames passing rules receive multiclass ML categorization with top-5 SHAP feature explanations. The legacy sequence-based DistilBERT MLM model is preserved as an offline research baseline.',
              },
            ].map((item) => (
              <div
                key={item.title}
                style={{
                  background: 'var(--color-bg-base)',
                  border: '1px solid var(--color-border)',
                  borderLeft: `3px solid ${item.borderColor}`,
                  borderRadius: '8px',
                  padding: '14px 16px',
                }}
              >
                <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: '8px' }}>
                  {item.title}
                </div>
                <p style={{ fontSize: '0.82rem', lineHeight: 1.65, color: 'var(--color-text-secondary)', margin: 0 }}>{item.text}</p>
              </div>
            ))}
          </div>
        </section>

        <section style={SECTION_CARD_STYLE}>
          <SectionHeader
            icon={<ShieldAlert size={16} color="var(--color-primary)" />}
            title="Security Notes & Operational Safeguards"
            subtitle="Limitations, deployment guidance, and responsible use"
          />

          <svg viewBox="0 0 700 60" xmlns="http://www.w3.org/2000/svg" style={{ width: '100%', height: 'auto', marginBottom: '16px' }}>
            <rect width="700" height="60" rx="8" fill="#FFFBEB" stroke="#D69E2E" strokeWidth="1.5" />

            <polygon points="36,42 50,18 64,42" fill="none" stroke="#D69E2E" strokeWidth="2" />
            <line x1="50" y1="26" x2="50" y2="34" stroke="#D69E2E" strokeWidth="2" />
            <circle cx="50" cy="38" r="1.5" fill="#D69E2E" />

            <text x="80" y="26" fontSize="11" fontFamily="Inter,sans-serif" fontWeight="700" fill="#B45309">
              Research and Educational Use Only
            </text>
            <text x="80" y="44" fontSize="9" fontFamily="Inter,sans-serif" fill="#92400E">
              Do not deploy against live vehicle networks without explicit written authorization. All testing must be performed in
              controlled laboratory environments.
            </text>
          </svg>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: '10px',
              marginTop: '16px',
              marginBottom: '16px',
            }}
          >
            {[
              {
                title: 'Before Deployment',
                danger: false,
                items: [
                  'Recalibrate threshold on target vehicle',
                  'Capture 5000+ normal sequences per condition',
                  'Test with known attack injections',
                  'Validate false alarm rate < 2%',
                  'Review vocab coverage for target ECUs',
                ],
              },
              {
                title: 'Runtime Safeguards',
                danger: false,
                items: [
                  'Require 3 consecutive anomalies before alert',
                  'Rate-limit /predict endpoint via nginx',
                  'Log all anomaly scores with timestamps',
                  'Authenticate WebSocket in production',
                  'Deploy behind TLS reverse proxy',
                ],
              },
              {
                title: 'Known Limitations',
                danger: true,
                items: [
                  'Single-vehicle training dataset',
                  'Vulnerable to low-and-slow attacks',
                  'Stage 2 limited to 4 known attack types',
                  'No real-time retraining capability',
                  'Memory-only alert log (lost on restart)',
                ],
              },
            ].map((checklist) => (
              <div
                key={checklist.title}
                style={{
                  background: checklist.danger ? 'var(--color-danger-bg)' : 'var(--color-bg-base)',
                  border: checklist.danger ? '1px solid rgba(196,30,58,0.2)' : '1px solid var(--color-border)',
                  borderRadius: '8px',
                  padding: '14px 16px',
                }}
              >
                <div
                  style={{
                    fontSize: '0.82rem',
                    fontWeight: 600,
                    color: checklist.danger ? 'var(--color-danger)' : 'var(--color-text-primary)',
                    marginBottom: '10px',
                  }}
                >
                  {checklist.title}
                </div>

                {checklist.items.map((item) => {
                  const isWarning = checklist.danger;
                  return (
                    <div
                      key={item}
                      style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: '8px',
                        marginBottom: '6px',
                      }}
                    >
                      <span
                        style={{
                          width: '16px',
                          height: '16px',
                          borderRadius: '50%',
                          fontSize: '9px',
                          fontWeight: 700,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                          marginTop: '1px',
                          background: isWarning ? '#D69E2E' : '#0F7B55',
                          color: 'white',
                        }}
                      >
                        {isWarning ? '⚠' : '✓'}
                      </span>
                      <span style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', lineHeight: 1.4 }}>{item}</span>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>

          <p style={{ ...BODY_TEXT_STYLE, marginTop: 0 }}>
            The current deployment includes baseline safeguards for transport security, request-level controls, and operational
            telemetry continuity. These controls improve reliability but are not a substitute for environment-specific hardening.
          </p>

          <div
            style={{
              marginTop: '12px',
              background: 'var(--color-bg-base)',
              border: '1px solid var(--color-border)',
              borderRadius: '8px',
              padding: '14px 16px',
            }}
          >
            <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: '8px' }}>
              Known Limitations
            </div>
            <p style={{ ...BODY_TEXT_STYLE, margin: 0 }}>
              The Stage 1 anomaly detector is trained only on normal traffic from the Car-Hacking dataset recorded from a single
              vehicle model. Deployment on a vehicle with a different ECU configuration will require retraining or threshold
              recalibration, as the normal CAN ID vocabulary and frequency patterns vary significantly between vehicle manufacturers and
              models.
            </p>
            <p style={{ ...BODY_TEXT_STYLE, marginTop: '10px', marginBottom: 0 }}>
              The system is vulnerable to low-and-slow adversarial attacks where a sophisticated attacker injects malicious frames at a
              rate slow enough to blend into normal traffic patterns. The sliding window size of 64 and 99th percentile threshold are
              tuned for the attack intensities present in the Car-Hacking dataset and may not generalise to subtler injection
              strategies.
            </p>
            <p style={{ ...BODY_TEXT_STYLE, marginTop: '10px', marginBottom: 0 }}>
              The Stage 2 classifier was trained on 1,000 windows per attack class. While this is sufficient for the four modelled
              attack types, the classifier has no fallback for novel attack types not seen during training. Unknown attack patterns will
              be detected by Stage 1 as anomalous but classified as the nearest known attack type by Stage 2.
            </p>
          </div>

          <div
            style={{
              marginTop: '12px',
              background: 'var(--color-bg-base)',
              border: '1px solid var(--color-border)',
              borderRadius: '8px',
              padding: '14px 16px',
            }}
          >
            <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: '8px' }}>
              Deployment Recommendations
            </div>
            <p style={{ ...BODY_TEXT_STYLE, margin: 0 }}>
              First, run threshold recalibration on traffic recorded from the specific target vehicle before deployment. Use a minimum
              of 5,000 normal sequences captured under diverse driving conditions (idle, city, highway, parking) to ensure the threshold
              reflects the full range of legitimate traffic patterns.
            </p>
            <p style={{ ...BODY_TEXT_STYLE, marginTop: '10px', marginBottom: 0 }}>
              Second, deploy the backend behind a reverse proxy such as Nginx with rate limiting on the /predict endpoint to prevent
              denial-of-service attacks against the detection system itself. The WebSocket endpoint should be authenticated in production
              environments.
            </p>
            <p style={{ ...BODY_TEXT_STYLE, marginTop: '10px', marginBottom: 0 }}>
              Third, implement alert aggregation logic before triggering safety-critical responses. Require at least three consecutive
              anomalous windows before escalating to a high-severity alert to reduce the operational impact of false positives.
            </p>
            <p style={{ ...BODY_TEXT_STYLE, marginTop: '10px', marginBottom: 0 }}>
              Fourth, log all anomaly scores with timestamps and sequence data to persistent storage for post-incident forensic analysis.
              Memory-only logs are lost on backend restart.
            </p>
          </div>

          <div
            style={{
              marginTop: '12px',
              background: 'var(--color-bg-base)',
              border: '1px solid var(--color-border)',
              borderRadius: '8px',
              padding: '14px 16px',
            }}
          >
            <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: '8px' }}>
              Adversarial Robustness
            </div>
            <p style={{ ...BODY_TEXT_STYLE, margin: 0 }}>
              The Masked Language Modelling objective provides inherent robustness against simple adversarial strategies. Because the
              model learns bidirectional contextual dependencies across all 64 positions simultaneously, an attacker cannot simply avoid
              a small set of blacklisted CAN IDs.
            </p>
            <p style={{ ...BODY_TEXT_STYLE, marginTop: '10px', marginBottom: 0 }}>
              However, an attacker with knowledge of the trained model could craft adversarial sequences that minimise reconstruction
              loss while still achieving an attack objective. The two-stage architecture adds resilience because Stage 2 evaluates a
              separate feature space based on TF-IDF and numeric behavior statistics.
            </p>
          </div>
        </section>

        <section style={SECTION_CARD_STYLE}>
          <SectionHeader
            icon={<Database size={16} color="var(--color-primary)" />}
            title="Data Pipeline"
            subtitle="From raw CAN logs to training-ready tensors"
          />

          <svg viewBox="0 0 700 80" xmlns="http://www.w3.org/2000/svg" style={{ width: '100%', height: 'auto', marginBottom: '16px' }}>
            <rect width="700" height="80" fill="#F4F6FB" rx="8" />

            <rect x="10" y="15" width="110" height="50" rx="6" fill="white" stroke="#E2E8F4" strokeWidth="1.5" />
            <text x="65" y="36" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#0F1729">
              Raw CSVs
            </text>
            <text x="65" y="49" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#8896A8">
              18.3M rows
            </text>
            <text x="65" y="60" textAnchor="middle" fontSize="7" fontFamily="Inter,sans-serif" fill="#8896A8">
              DoS/Fuzzy/RPM/Gear
            </text>

            <line x1="120" y1="40" x2="138" y2="40" stroke="#1A6FD4" strokeWidth="1.5" markerEnd="url(#arrowDP)" />

            <rect x="140" y="15" width="110" height="50" rx="6" fill="white" stroke="#E2E8F4" strokeWidth="1.5" />
            <text x="195" y="36" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#0F1729">
              Augment
            </text>
            <text x="195" y="49" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#8896A8">
              56.4M rows
            </text>
            <text x="195" y="60" textAnchor="middle" fontSize="7" fontFamily="Inter,sans-serif" fill="#8896A8">
              4x expansion
            </text>

            <line x1="250" y1="40" x2="268" y2="40" stroke="#1A6FD4" strokeWidth="1.5" markerEnd="url(#arrowDP)" />

            <rect x="270" y="15" width="110" height="50" rx="6" fill="white" stroke="#E2E8F4" strokeWidth="1.5" />
            <text x="325" y="36" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#0F1729">
              Tokenize
            </text>
            <text x="325" y="49" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#8896A8">
              vocab.json
            </text>
            <text x="325" y="60" textAnchor="middle" fontSize="7" fontFamily="Inter,sans-serif" fill="#8896A8">
              hex to integer
            </text>

            <line x1="380" y1="40" x2="398" y2="40" stroke="#1A6FD4" strokeWidth="1.5" markerEnd="url(#arrowDP)" />

            <rect x="400" y="15" width="110" height="50" rx="6" fill="white" stroke="#E2E8F4" strokeWidth="1.5" />
            <text x="455" y="36" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#0F1729">
              Window
            </text>
            <text x="455" y="49" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#8896A8">
              500K seqs
            </text>
            <text x="455" y="60" textAnchor="middle" fontSize="7" fontFamily="Inter,sans-serif" fill="#8896A8">
              size=64 step=32
            </text>

            <line x1="510" y1="40" x2="528" y2="40" stroke="#1A6FD4" strokeWidth="1.5" markerEnd="url(#arrowDP)" />

            <rect x="530" y="15" width="160" height="50" rx="6" fill="#EBF4FF" stroke="#1A6FD4" strokeWidth="1.5" />
            <text x="610" y="36" textAnchor="middle" fontSize="9" fontFamily="Inter,sans-serif" fontWeight="600" fill="#1A6FD4">
              sequences.pt
            </text>
            <text x="610" y="49" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono,monospace" fill="#1A6FD4">
              tensor(500000, 64)
            </text>
            <text x="610" y="60" textAnchor="middle" fontSize="7" fontFamily="Inter,sans-serif" fill="#1A6FD4">
              ready for training
            </text>

            <defs>
              <marker id="arrowDP" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                <path d="M0,0 L0,6 L6,3 z" fill="#1A6FD4" />
              </marker>
            </defs>
          </svg>

          <table
            style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: '0.78rem',
            }}
          >
            <thead>
              <tr style={{ background: 'var(--color-bg-base)' }}>
                {['Dataset File', 'Attack Type', 'Approx Rows', 'Flag'].map((heading) => (
                  <th
                    key={heading}
                    style={{
                      padding: '10px 14px',
                      textAlign: 'left',
                      fontSize: '0.68rem',
                      fontWeight: 600,
                      textTransform: 'uppercase',
                      letterSpacing: '0.08em',
                      color: 'var(--color-text-muted)',
                      borderBottom: '2px solid var(--color-border)',
                    }}
                  >
                    {heading}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ['DoS_dataset.csv', 'DoS Flooding', '~3.6M', 'T'],
                ['Fuzzy_dataset.csv', 'Fuzzy Injection', '~3.7M', 'T'],
                ['RPM_dataset.csv', 'RPM Spoofing', '~4.0M', 'T'],
                ['gear_dataset.csv', 'Gear Spoofing', '~4.0M', 'T'],
                ['normal_run_data.csv', 'Normal Driving', '~3.0M', 'R'],
              ].map(([file, attack, rows, flag], index) => (
                <tr key={file} style={{ background: index % 2 === 0 ? 'var(--color-bg-base)' : 'white' }}>
                  <td
                    style={{
                      padding: '9px 14px',
                      borderBottom: '1px solid var(--color-border)',
                      fontSize: '0.78rem',
                      color: 'var(--color-text-secondary)',
                      fontFamily: 'JetBrains Mono, monospace',
                    }}
                  >
                    {file}
                  </td>
                  <td
                    style={{
                      padding: '9px 14px',
                      borderBottom: '1px solid var(--color-border)',
                      fontSize: '0.78rem',
                      color: 'var(--color-text-secondary)',
                    }}
                  >
                    {attack}
                  </td>
                  <td
                    style={{
                      padding: '9px 14px',
                      borderBottom: '1px solid var(--color-border)',
                      fontSize: '0.78rem',
                      color: 'var(--color-text-secondary)',
                      fontFamily: 'JetBrains Mono, monospace',
                    }}
                  >
                    {rows}
                  </td>
                  <td
                    style={{
                      padding: '9px 14px',
                      borderBottom: '1px solid var(--color-border)',
                      fontSize: '0.75rem',
                      fontFamily: 'JetBrains Mono, monospace',
                      color: flag === 'T' ? 'var(--color-danger)' : 'var(--color-success)',
                      fontWeight: 600,
                    }}
                  >
                    {flag}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      </div>
    </AppLayout>
  );
}

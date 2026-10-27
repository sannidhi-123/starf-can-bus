import { Link } from 'react-router-dom';

export function HomePage() {
  return (
    <div className="bg-surface text-on-surface selection:bg-secondary-container selection:text-on-secondary-container">
      <nav className="fixed top-0 z-50 w-full bg-surface-container-low/80 backdrop-blur-md">
        <div className="mx-auto flex h-16 w-full max-w-[1440px] items-center justify-between px-6">
          <div className="flex items-center gap-2">
            <span className="font-headline text-xl font-bold tracking-tighter text-primary">STARF</span>
          </div>
          <div className="hidden items-center gap-8 md:flex">
            <Link className="border-b-2 border-primary pb-1 text-sm font-bold text-primary" to="/">
              Home
            </Link>
            <Link className="text-sm text-on-surface-variant transition-colors hover:text-primary" to="/security">
              About
            </Link>
            <Link className="text-sm text-on-surface-variant transition-colors hover:text-primary" to="/reports">
              Reports
            </Link>
          </div>
          <div className="flex items-center gap-4">
            <button className="rounded-full p-2 transition-all hover:bg-surface-container-highest active:scale-95" type="button">
              <span className="material-symbols-outlined text-on-surface-variant">search</span>
            </button>
            <Link
              className="hidden items-center gap-2 rounded-xl bg-primary px-4 py-2 text-sm font-bold text-on-primary transition-all hover:-translate-y-[1px] active:scale-95 md:flex"
              to="/dashboard"
            >
              Live Dashboard
            </Link>
            <button className="p-2 md:hidden" type="button" aria-label="Open menu">
              <span className="material-symbols-outlined">menu</span>
            </button>
          </div>
        </div>
      </nav>

      <main className="pt-16">
        <section className="relative flex min-h-[870px] flex-col items-center justify-center overflow-hidden px-6">
          <div className="hex-grid absolute inset-0 z-0" />
          <div className="absolute left-1/2 top-1/2 z-0 h-[600px] w-[600px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-secondary/5 blur-[120px]" />
          <div className="relative z-10 max-w-4xl text-center">
            <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-secondary/10 bg-secondary-container/20 px-3 py-1 text-secondary">
              <span className="h-2 w-2 animate-pulse rounded-full bg-secondary" />
              <span className="font-mono text-xs font-bold uppercase tracking-widest">System Status: Active</span>
            </div>
            <h1 className="mb-6 font-headline text-4xl font-bold leading-[1.15] tracking-tight text-primary md:text-6xl">
              STARF <span className="block text-2xl font-semibold text-primary/80 md:text-4xl mt-2">(Smart Traffic Anomaly Recognition Framework)</span>
            </h1>
            <p className="mx-auto mb-10 max-w-3xl text-xl leading-relaxed text-on-surface-variant md:text-2xl">
              An Explainable Hybrid Anomaly Detection System for Intra-Vehicular CAN-Bus Networks
            </p>
            <div className="flex flex-col items-center justify-center gap-4 md:flex-row">
              <Link
                className="w-full rounded-xl bg-primary px-8 py-4 font-bold text-white shadow-lg shadow-primary/10 transition-all hover:-translate-y-0.5 hover:shadow-primary/20 md:w-auto"
                to="/dashboard"
              >
                View Live Dashboard
              </Link>
              <Link
                className="w-full rounded-xl border-2 border-outline/20 bg-transparent px-8 py-4 font-bold text-primary transition-all hover:bg-surface-container-low md:w-auto"
                to="/integration"
              >
                Try Sequence Analyzer
              </Link>
            </div>
          </div>

          <div className="group relative mt-20 w-full max-w-5xl">
            <div className="overflow-hidden rounded-2xl border border-outline-variant/30 bg-surface-container-lowest shadow-2xl">
              <div className="flex items-center gap-2 bg-surface-container-low px-4 py-3">
                <div className="flex gap-1.5">
                  <div className="h-3 w-3 rounded-full bg-error/20" />
                  <div className="h-3 w-3 rounded-full bg-surface-tint/20" />
                  <div className="h-3 w-3 rounded-full bg-tertiary-container/20" />
                </div>
                <div className="mx-auto font-mono text-[10px] uppercase tracking-widest text-outline">STARF — System Architecture Overview</div>
              </div>
              <img
                alt="Architectural diagram of the STARF Hybrid CAN Bus Intrusion Detection System showing ECU nodes, CAN Bus protocol line, and the anomaly detection module"
                className="h-[400px] w-full object-cover"
                src="/ids_architecture.png"
              />
            </div>
          </div>
        </section>

        <section className="border-y border-outline-variant/10 bg-surface-container-low py-12">
          <div className="mx-auto max-w-[1440px] px-6">
            <div className="grid grid-cols-1 items-center gap-12 md:grid-cols-2">
              <div className="flex flex-wrap items-center justify-center gap-8 md:justify-start">
                <div className="text-center md:text-left">
                  <div className="font-headline text-3xl font-bold text-primary">97.12%</div>
                  <div className="font-mono text-[10px] uppercase tracking-widest text-outline">Binary Detection Accuracy</div>
                </div>
                <div className="text-center md:text-left">
                  <div className="font-headline text-3xl font-bold text-primary">97.53%</div>
                  <div className="font-mono text-[10px] uppercase tracking-widest text-outline">Precision (Stage-1)</div>
                </div>
                <div className="text-center md:text-left">
                  <div className="font-headline text-3xl font-bold text-primary">95.06%</div>
                  <div className="font-mono text-[10px] uppercase tracking-widest text-outline">F1 Score (Weighted)</div>
                </div>
              </div>
              <div className="flex flex-wrap justify-center gap-6 opacity-60 md:justify-end">
                <span className="font-headline font-bold text-on-surface-variant">PyTorch</span>
                <span className="font-headline font-bold text-on-surface-variant">Transformers</span>
                <span className="font-headline font-bold text-on-surface-variant">scikit-learn</span>
                <span className="font-headline font-bold text-on-surface-variant">FastAPI</span>
                <span className="font-headline font-bold text-on-surface-variant">React</span>
              </div>
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-[1440px] overflow-hidden px-6 py-24">
          <div className="grid grid-cols-1 items-center gap-16 lg:grid-cols-2">
            <div className="relative">
              <div className="absolute -left-12 -top-12 h-64 w-64 rounded-full bg-error/5 blur-3xl" />
              <h2 className="relative z-10 mb-8 font-headline text-4xl font-bold text-primary">Structurally Unencrypted &amp; Inherently Vulnerable</h2>
              <p className="mb-6 text-lg leading-relaxed text-on-surface-variant">
                The Controller Area Network (CAN) protocol, standardized as ISO 11898, was architecturally designed without inherent authentication, encryption, or access control mechanisms. As in-vehicle connectivity has proliferated through V2X and OBD-II interfaces, this foundational absence of security primitives has been widely documented as a critical attack surface.
              </p>
              <p className="text-lg leading-relaxed text-on-surface-variant">
                A compromised Electronic Control Unit (ECU) is capable of broadcasting arbitrarily crafted frames onto the shared bus, potentially enabling unauthorized manipulation of safety-critical subsystems including steering, braking, and powertrain control. Traditional perimeter-based firewalls are demonstrated to be insufficient, as they fail to capture the semantic and temporal structure of legitimate CAN bus grammar.
              </p>
            </div>
            <div className="relative rounded-2xl border border-outline-variant/20 bg-surface-container-low p-8">
              <div className="mb-4 flex items-center gap-2 font-mono text-xs text-error">
                <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>
                  warning
                </span>
                ALERT: UNAUTHORIZED FRAME INJECTION DETECTED
              </div>
              <div className="space-y-2 font-mono text-[13px]">
                <div className="flex justify-between rounded bg-error-container/20 px-2 py-1 text-error">
                  <span>ID: 0x2B0</span>
                  <span>DATA: 01 55 22 00 00 00 00 00</span>
                  <span>[FUZZY ATTACK]</span>
                </div>
                <div className="flex justify-between px-2 py-1 text-outline-variant">
                  <span>ID: 0x18F</span>
                  <span>DATA: 00 00 00 4F 21 00 00 00</span>
                  <span>[NOMINAL]</span>
                </div>
                <div className="flex justify-between px-2 py-1 text-outline-variant">
                  <span>ID: 0x316</span>
                  <span>DATA: 05 21 00 00 00 00 00 00</span>
                  <span>[NOMINAL]</span>
                </div>
                <div className="flex justify-between rounded bg-error-container/20 px-2 py-1 text-error">
                  <span>ID: 0x2B0</span>
                  <span>DATA: FF FF FF FF FF FF FF FF</span>
                  <span>[DoS ATTACK]</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="relative overflow-hidden bg-primary py-24 text-white">
          <div className="hex-grid absolute inset-0 opacity-10" />
          <div className="relative z-10 mx-auto max-w-[1440px] px-6">
            <div className="mb-20 max-w-3xl">
              <h2 className="mb-8 font-headline text-4xl font-bold">Dual-Stage Intelligence Architecture</h2>
              <p className="text-lg leading-relaxed text-on-primary-container">
                The STARF platform implements an explainable hybrid anomaly detection pipeline. In the first stage, deterministic automotive rule heuristics (arbitration ID whitelist, transmission frequency window, and inter-arrival timing checks) catch known and anomalous traffic patterns with 0ms overhead. Frames passing rule validation are subsequently evaluated by an ensemble Random Forest classifier with SHAP explainability for granular threat taxonomy categorization: DoS, Fuzzy Injection, Gear Spoofing, and RPM Manipulation.
              </p>
            </div>

            <div className="relative grid grid-cols-1 items-center gap-4 md:grid-cols-4 md:gap-8">
              <div className="relative rounded-xl border border-white/10 bg-primary-container p-6 text-center">
                <div className="mb-2 font-mono text-[10px] uppercase tracking-widest text-on-primary-container">Input Layer</div>
                <div className="font-headline text-lg font-bold">Raw CAN Telemetry</div>
                <div className="mt-4 flex justify-center gap-1">
                  <div className="h-4 w-1 bg-secondary" />
                  <div className="h-6 w-1 bg-secondary" />
                  <div className="h-3 w-1 bg-secondary" />
                </div>
              </div>
              <div className="relative rounded-xl border border-white/10 bg-primary-container p-6 text-center">
                <div className="mb-2 font-mono text-[10px] uppercase tracking-widest text-on-primary-container">Stage 1: Anomaly Scorer</div>
                <div className="font-headline text-lg font-bold">DistilBERT MLM</div>
                <div className="mt-4 text-[10px] text-on-primary-container">Contextual Embedding &amp; Reconstruction Loss</div>
              </div>
              <div className="relative rounded-xl border border-white/10 bg-primary-container p-6 text-center">
                <div className="mb-2 font-mono text-[10px] uppercase tracking-widest text-on-primary-container">Stage 2: Threat Classifier</div>
                <div className="font-headline text-lg font-bold">Random Forest (128 Estimators)</div>
                <div className="mt-4 text-[10px] text-on-primary-container">Multiclass Attack Taxonomy Decision</div>
              </div>
              <div className="relative rounded-xl border border-white/10 bg-secondary p-6 text-center">
                <div className="mb-2 font-mono text-[10px] uppercase tracking-widest text-primary">Output</div>
                <div className="font-headline text-lg font-bold text-primary">SOC Alert Emission</div>
                <div className="mt-4 text-[10px] text-primary">Real-time Threat Mitigation &amp; Reporting</div>
              </div>
            </div>
          </div>
        </section>

        <section className="mx-auto max-w-[1440px] px-6 py-24">
          <h2 className="mb-16 text-center font-headline text-4xl font-bold text-primary">Precision Infrastructure</h2>
          <div className="grid grid-cols-1 gap-6 md:grid-cols-12">
            <div className="col-span-12 flex flex-col items-center gap-8 rounded-2xl border border-outline-variant/10 bg-surface-container-low p-8 transition-all hover:border-secondary/30 md:col-span-8 md:flex-row">
              <div className="flex-1">
                <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-xl bg-secondary/10 text-secondary">
                  <span className="material-symbols-outlined">bolt</span>
                </div>
                <h3 className="mb-4 font-headline text-2xl font-bold text-primary">Sub-Millisecond Detection Latency</h3>
                <p className="text-on-surface-variant">
                  Anomalous sequences are detected and classified within 12ms of frame ingestion through persistent WebSocket telemetry channels. The system maintains continuous, non-blocking inference throughput, ensuring that no malicious frame injection event is permitted to propagate undetected across the CAN bus arbitration window.
                </p>
              </div>
              <div className="aspect-video w-full overflow-hidden rounded-xl border border-outline-variant/20 bg-surface-container-lowest md:w-64">
                <img
                  alt="Real-time anomaly detection telemetry stream visualization"
                  className="h-full w-full object-cover"
                  src="/ids_architecture.png"
                />
              </div>
            </div>

            <div className="col-span-12 rounded-2xl border border-outline-variant/10 bg-surface-container-low p-8 transition-all hover:border-secondary/30 md:col-span-4">
              <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-xl bg-secondary/10 text-secondary">
                <span className="material-symbols-outlined">category</span>
              </div>
              <h3 className="mb-4 font-headline text-2xl font-bold text-primary">Multiclass Threat Taxonomy</h3>
              <p className="mb-6 text-on-surface-variant">Granular Stage-2 classification of four distinct attack vector categories is performed, achieving 100% accuracy on the evaluation partition (noted as overfitted performance requiring further generalization validation).</p>
              <div className="space-y-2">
                <div className="flex items-center gap-3">
                  <div className="h-2 w-2 rounded-full bg-secondary" />
                  <span className="font-mono text-xs text-on-surface-variant">Denial of Service (Bus Flooding)</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="h-2 w-2 rounded-full bg-secondary" />
                  <span className="font-mono text-xs text-on-surface-variant">Fuzzy Injection (Random Payload)</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="h-2 w-2 rounded-full bg-secondary" />
                  <span className="font-mono text-xs text-on-surface-variant">Gear Spoofing (Actuator Manipulation)</span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="h-2 w-2 rounded-full bg-secondary" />
                  <span className="font-mono text-xs text-on-surface-variant">RPM Spoofing (Tachometer Falsification)</span>
                </div>
              </div>
            </div>

            <div className="col-span-12 rounded-2xl border border-outline-variant/10 bg-surface-container-low p-8 transition-all hover:border-secondary/30 md:col-span-4">
              <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-xl bg-secondary/10 text-secondary">
                <span className="material-symbols-outlined">api</span>
              </div>
              <h3 className="mb-4 font-headline text-2xl font-bold text-primary">Production-Grade API</h3>
              <p className="text-on-surface-variant">
                The inference backend is architected on the FastAPI framework, providing asynchronous RESTful endpoints and persistent WebSocket channels optimized for high-throughput integration with OEM telematics platforms and fleet management consoles.
              </p>
            </div>

            <div className="col-span-12 flex flex-col items-center gap-8 rounded-2xl border border-outline-variant/10 bg-surface-container-low p-8 transition-all hover:border-secondary/30 md:col-span-8 md:flex-row-reverse">
              <div className="flex-1">
                <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-xl bg-secondary/10 text-secondary">
                  <span className="material-symbols-outlined">analytics</span>
                </div>
                <h3 className="mb-4 font-headline text-2xl font-bold text-primary">Comprehensive Forensic Analytics</h3>
                <p className="text-on-surface-variant">
                  Interactive observability dashboards facilitate deep-dive analysis of network health metrics, per-sequence anomaly score distributions, attack vector frequency histograms, and predictive maintenance indicators for secure automotive operations.
                </p>
              </div>
              <div className="aspect-video w-full overflow-hidden rounded-xl border border-outline-variant/20 bg-surface-container-lowest md:w-64">
                <img
                  alt="Forensic analytics dashboard with anomaly score distribution and attack vector classification"
                  className="h-full w-full object-cover"
                  src="/ids_architecture.png"
                />
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="mt-auto w-full bg-surface-container-low">
        <div className="mx-auto flex w-full max-w-[1440px] flex-col items-center justify-between px-8 py-12 md:flex-row">
          <div className="mb-8 md:mb-0">
            <div className="mb-2 font-headline text-xl font-bold text-primary">STARF</div>
            <div className="text-xs text-on-surface-variant">© 2026 STARF — Explainable Hybrid Anomaly Detection System for Intra-Vehicular CAN-Bus Networks</div>
          </div>
          <div className="flex gap-12">
            <div className="flex flex-col gap-2">
              <span className="mb-2 font-mono text-[10px] uppercase tracking-widest text-outline">Platform</span>
              <Link className="text-xs text-on-surface-variant transition-colors hover:text-primary" to="/security">
                Mission
              </Link>
              <Link className="text-xs text-on-surface-variant transition-colors hover:text-primary" to="/pipeline">
                Tech Stack
              </Link>
            </div>
            <div className="flex flex-col gap-2">
              <span className="mb-2 font-mono text-[10px] uppercase tracking-widest text-outline">Resources</span>
              <Link className="text-xs text-on-surface-variant underline transition-colors hover:text-primary" to="/integration">
                Documentation
              </Link>
              <Link className="text-xs text-on-surface-variant transition-colors hover:text-primary" to="/security">
                Contact
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

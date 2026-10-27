import { ConsoleLayout } from '../layout/ConsoleLayout';

export function PipelinePage() {
  return (
    <ConsoleLayout activeNav="pipeline">
      <main className="min-h-screen p-8">
        <header className="mb-12 max-w-4xl">
          <h1 className="mb-2 font-headline text-4xl font-bold tracking-tight text-primary">Model &amp; Pipeline Architecture</h1>
          <p className="text-lg text-on-surface-variant">Formal specification and forensic visualization of the dual-stage inference pipeline, artifact lineage, and decision logic governing the STARF anomaly detection framework.</p>
        </header>

        {/* Section 1: Abstract */}
        <section className="mb-16 rounded-2xl border border-outline-variant/10 bg-surface-container-low p-8 md:p-10">
          <div className="mb-2 font-mono text-[10px] font-bold uppercase tracking-widest text-secondary">Section 1</div>
          <h2 className="mb-6 font-headline text-2xl font-bold text-primary">Abstract</h2>
          <p className="mb-4 leading-relaxed text-on-surface-variant">
            A dual-stage Hybrid Intrusion Detection System (STARF — Smart Traffic Anomaly Recognition Kit) is presented for real-time identification and classification of cyber-attacks targeting the Controller Area Network (CAN) bus protocol. The operational architecture integrates deterministic automotive rule heuristics (Stage 1 arbitration ID whitelist, frequency, and inter-arrival timing monitors) with a frame-level multiclass Random Forest ML classifier (Stage 2) trained with SMOTE balancing. The sequence-based DistilBERT Transformer model trained via Masked Language Modeling (MLM) serves as an offline research baseline.
          </p>
          <p className="leading-relaxed text-on-surface-variant">
            On the rigorous held-out chronological 80/20 test evaluation (100,000 frames across 5 classes), the operational STARF Hybrid Pipeline achieves a detection <strong>Precision of 99.68%</strong>, <strong>Recall of 100.00%</strong>, and <strong>F1-score of 99.84%</strong>, completely catching all DoS, Fuzzy, Spoof_Gear, and Spoof_RPM injection frames with zero false negatives.
          </p>
        </section>

        {/* Section 2: Methodology */}
        <section className="mb-16 rounded-2xl border border-outline-variant/10 bg-surface-container-low p-8 md:p-10">
          <div className="mb-2 font-mono text-[10px] font-bold uppercase tracking-widest text-secondary">Section 2</div>
          <h2 className="mb-6 font-headline text-2xl font-bold text-primary">Methodology</h2>
          <p className="mb-6 leading-relaxed text-on-surface-variant">
            The inference methodology is decomposed into two sequential stages, connected by a threshold gating mechanism. Raw CAN bus frames are first normalized and aggregated into fixed-length sliding windows of 64 arbitration identifiers. Each window is then processed through the dual-stage pipeline as follows:
          </p>

          <div className="mb-2 flex items-center justify-between px-4">
            <h3 className="font-headline text-xl font-bold">Inference Pipeline Flow</h3>
            <span className="rounded-full bg-tertiary-container px-3 py-1 font-label text-xs text-tertiary-fixed-dim">STATUS: ACTIVE</span>
          </div>

          <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-5">
            <div className="group relative flex flex-col items-center overflow-hidden rounded-xl bg-surface-container-lowest p-6 text-center">
              <div className="absolute left-0 top-0 h-full w-1 bg-secondary" />
              <span className="material-symbols-outlined mb-3 text-3xl text-secondary">settings_input_component</span>
              <h4 className="mb-1 font-headline text-sm font-bold">Data Normalization</h4>
              <p className="text-xs text-on-surface-variant">CAN frame hex ID canonicalization and bus noise filtering</p>
            </div>

            <div className="relative flex flex-col items-center overflow-hidden rounded-xl bg-surface-container-lowest p-6 text-center">
              <div className="absolute left-0 top-0 h-full w-1 bg-secondary" />
              <span className="material-symbols-outlined mb-3 text-3xl text-secondary">view_column</span>
              <h4 className="mb-1 font-headline text-sm font-bold">Sequence Tokenization</h4>
              <p className="text-xs text-on-surface-variant">
                Fixed-length sliding window: <span className="font-label">64</span> identifiers per sequence
              </p>
            </div>

            <div className="relative flex flex-col items-center rounded-xl bg-primary-container p-6 text-center text-on-primary-container">
              <div className="absolute -right-2 -top-2 rounded-full bg-secondary-container px-2 py-0.5 text-[10px] font-bold text-on-secondary-container">STAGE 1</div>
              <span className="material-symbols-outlined mb-3 text-3xl text-secondary-container">psychology</span>
              <h4 className="mb-1 font-headline text-sm font-bold">Anomaly Scoring Engine</h4>
              <p className="text-xs opacity-80">DistilBERT MLM reconstruction loss quantification</p>
            </div>

            <div className="relative flex flex-col items-center rounded-xl bg-surface-container-lowest p-6 text-center">
              <span className="material-symbols-outlined mb-3 text-3xl text-on-surface-variant">door_front</span>
              <h4 className="mb-1 font-headline text-sm font-bold">Threshold Gate (τ)</h4>
              <p className="text-xs text-on-surface-variant">Empirically calibrated decision boundary (99th percentile)</p>
            </div>

            <div className="relative flex flex-col items-center rounded-xl bg-primary p-6 text-center text-on-primary">
              <div className="absolute -right-2 -top-2 rounded-full bg-secondary-container px-2 py-0.5 text-[10px] font-bold text-on-secondary-container">STAGE 2</div>
              <span className="material-symbols-outlined mb-3 text-3xl text-secondary-fixed-dim">hub</span>
              <h4 className="mb-1 font-headline text-sm font-bold text-white">Multiclass Classifier</h4>
              <p className="text-xs opacity-70">Random Forest ensemble (128 estimators) — DoS/Fuzzy/Gear/RPM taxonomy</p>
            </div>
          </div>
        </section>

        {/* Model Detail Cards */}
        <section className="mb-16 grid grid-cols-1 gap-8 lg:grid-cols-2">
          <div className="group relative flex flex-col gap-6 overflow-hidden rounded-xl bg-surface-container-lowest p-8">
            <div className="flex items-start justify-between">
              <div>
                <div className="mb-2 text-xs uppercase tracking-widest text-secondary">Core Scoring Unit — Stage 1</div>
                <h3 className="font-headline text-2xl font-bold text-primary">DistilBERT Anomaly Scorer</h3>
              </div>
              <span className="material-symbols-outlined text-4xl text-primary/10 transition-colors group-hover:text-primary/20">memory</span>
            </div>
            <div className="space-y-4">
              <p className="leading-relaxed text-on-surface-variant">
                A specialized Transformer-based encoder architecture, derived from the DistilBERT knowledge-distilled language model, is employed for sequential anomaly detection on automotive bus protocols. The model is pre-trained via Masked Language Modeling (MLM) on benign CAN arbitration ID sequences, thereby learning the implicit syntactic grammar of legitimate network communications. During inference, the reconstruction loss (cross-entropy) for each input sequence is computed and utilized as a continuous anomaly score. Elevated scores indicate statistically significant deviations from the learned baseline grammar, signaling potential intrusion activity.
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-lg bg-surface-container-low p-3">
                  <div className="mb-1 text-[10px] text-on-surface-variant">ARCHITECTURE</div>
                  <div className="text-sm font-bold">DistilBERT (6-layer Transformer)</div>
                </div>
                <div className="rounded-lg bg-surface-container-low p-3">
                  <div className="mb-1 text-[10px] text-on-surface-variant">TRAINING OBJECTIVE</div>
                  <div className="text-sm font-bold">Masked Language Modeling (MLM)</div>
                </div>
                <div className="rounded-lg bg-surface-container-low p-3">
                  <div className="mb-1 text-[10px] text-on-surface-variant">INPUT SPECIFICATION</div>
                  <div className="text-sm font-bold">64-ID sliding sequence window</div>
                </div>
                <div className="rounded-lg bg-surface-container-low p-3">
                  <div className="mb-1 text-[10px] text-on-surface-variant">INFERENCE LATENCY</div>
                  <div className="text-sm font-bold">&lt; 12ms per sequence (CPU)</div>
                </div>
              </div>
            </div>
          </div>

          <div className="group relative flex flex-col gap-6 overflow-hidden rounded-xl bg-surface-container-lowest p-8">
            <div className="flex items-start justify-between">
              <div>
                <div className="mb-2 text-xs uppercase tracking-widest text-secondary">Identification Unit — Stage 2</div>
                <h3 className="font-headline text-2xl font-bold text-primary">Random Forest Classifier</h3>
              </div>
              <span className="material-symbols-outlined text-4xl text-primary/10 transition-colors group-hover:text-primary/20">account_tree</span>
            </div>
            <div className="space-y-4">
              <p className="leading-relaxed text-on-surface-variant">
                The secondary classification stage is conditionally invoked when the Stage-1 anomaly score exceeds the calibrated detection threshold (τ). A Random Forest ensemble comprising 128 decision tree estimators is employed to provide high-granularity multiclass categorization of the detected attack vector. Feature extraction is performed via TF-IDF vectorization of the CAN ID sequence, supplemented by numeric sequence position features.
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-lg bg-surface-container-low p-3">
                  <div className="mb-1 text-[10px] text-on-surface-variant">FEATURE EXTRACTION</div>
                  <div className="text-sm font-bold">TF-IDF + Positional Encoding</div>
                </div>
                <div className="rounded-lg bg-surface-container-low p-3">
                  <div className="mb-1 text-[10px] text-on-surface-variant">OUTPUT TAXONOMY</div>
                  <div className="text-sm font-bold">4-Class (DoS/Fuzzy/Gear/RPM)</div>
                </div>
                <div className="rounded-lg bg-surface-container-low p-3">
                  <div className="mb-1 text-[10px] text-on-surface-variant">ENSEMBLE SIZE</div>
                  <div className="text-sm font-bold">128 Decision Trees</div>
                </div>
                <div className="rounded-lg bg-surface-container-low p-3">
                  <div className="mb-1 text-[10px] text-on-surface-variant">EVALUATION ACCURACY</div>
                  <div className="text-sm font-bold">100% F1 (Overfitted — See §3)</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Decision Logic */}
        <section className="mb-16 rounded-2xl border border-outline-variant/10 bg-surface-container-low p-10">
          <h3 className="mb-8 flex items-center gap-3 font-headline text-2xl font-bold">
            <span className="material-symbols-outlined text-secondary">alt_route</span>
            Decision Logic — Threshold Gating Mechanism
          </h3>

          <div className="flex flex-col items-center justify-between gap-8 md:flex-row">
            <div className="flex-1 space-y-4">
              <div className="flex items-center gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary font-label text-white">S1</div>
                <div className="flex-1 rounded-xl bg-surface-container-lowest p-4">
                  <span className="text-xs opacity-60">DISTILBERT RECONSTRUCTION LOSS (ρ)</span>
                  <div className="font-bold">ρ = 0.88 (Anomalous — Exceeds Threshold)</div>
                </div>
              </div>
              <div className="flex h-12 justify-center">
                <div className="w-px border-l border-dashed border-outline bg-outline-variant" />
              </div>
              <div className="flex items-center gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-full border-2 border-secondary text-secondary">
                  <span className="material-symbols-outlined">priority_high</span>
                </div>
                <div className="flex-1 rounded-xl bg-secondary-container p-4">
                  <span className="text-xs text-on-secondary-container/60">THRESHOLD GATE (τ = 99th Percentile)</span>
                  <div className="font-bold text-on-secondary-container">If ρ &gt; τ → Route to Stage-2 Multiclass Classifier</div>
                </div>
              </div>
            </div>

            <div className="hidden md:block">
              <span className="material-symbols-outlined text-5xl text-outline-variant">trending_flat</span>
            </div>

            <div className="flex-1 space-y-4">
              <div className="rounded-2xl border-l-4 border-error bg-surface-container-lowest p-6 shadow-sm">
                <h4 className="mb-4 font-headline text-sm font-bold">Stage-2: Multiclass Threat Categorization</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span>Denial of Service (Bus Flooding)</span>
                    <span className="font-label font-bold text-error">88%</span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-container-high">
                    <div className="h-full bg-error" style={{ width: '88%' }} />
                  </div>
                  <div className="flex items-center justify-between text-xs opacity-50">
                    <span>Fuzzy Injection (Random Payload)</span>
                    <span className="font-label">7%</span>
                  </div>
                  <div className="flex items-center justify-between text-xs opacity-50">
                    <span>Gear Spoofing (Actuator Manipulation)</span>
                    <span className="font-label">5%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: Experimental Setup */}
        <section className="mb-16 rounded-2xl border border-outline-variant/10 bg-surface-container-low p-8 md:p-10">
          <div className="mb-2 font-mono text-[10px] font-bold uppercase tracking-widest text-secondary">Section 3</div>
          <h2 className="mb-6 font-headline text-2xl font-bold text-primary">Experimental Setup</h2>
          <div className="mb-6 space-y-4 leading-relaxed text-on-surface-variant">
            <p>
              The system was evaluated on the Car-Hacking Dataset (Seo et al., 2018), a widely cited benchmark corpus comprising authentic CAN bus telemetry captured from a production vehicle. The dataset encompasses four distinct attack categories — Denial of Service (DoS), Fuzzy Injection, Gear Spoofing, and RPM Manipulation — interleaved with normal traffic sequences.
            </p>
            <p>
              A total of <strong>57,277 sequences</strong> were constructed using a 64-frame sliding window, of which <strong>40,149 sequences</strong> were labeled as normal and <strong>17,128 sequences</strong> as attack. The Stage-1 anomaly threshold was empirically calibrated at the 99th percentile of the normal score distribution, yielding a decision boundary of τ = 3.0 (with mean normal score μ = 0.583 and standard deviation σ = 1.041).
            </p>
            <p>
              Stage-2 multiclass classification achieves 100% accuracy on the current evaluation partition across all four attack categories. It is explicitly acknowledged that this performance metric represents overfitted behavior on the training data distribution. Cross-dataset validation on the OTIDS benchmark and synthetic CAN corpora is recommended prior to deployment in production automotive security environments.
            </p>
          </div>
        </section>

        {/* Artifact Registry */}
        <section className="mb-16">
          <h3 className="mb-8 font-headline text-2xl font-bold">Model Artifact Registry</h3>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
            {[
              ['book_4', 'CAN ID Vocabulary', 'HEX_CORPUS_V3.JSON'],
              ['layers', 'Sequence Tensor Buffer', 'TENSOR_BUFF_64.NPZ'],
              ['database', 'DistilBERT Weights', 'BEST_MODEL.PT (39MB)'],
              ['tune', 'Threshold Configuration', 'THRESHOLD.JSON (τ=3.0)'],
              ['data_object', 'TF-IDF Vectorizer', 'VECTORIZER.PKL'],
            ].map(([icon, title, file]) => (
              <div key={title} className="group flex flex-col gap-3 rounded-xl bg-surface-container-lowest p-5 transition-colors hover:bg-surface-container-high">
                <span className="material-symbols-outlined text-secondary">{icon}</span>
                <div>
                  <h5 className="text-sm font-bold">{title}</h5>
                  <p className="font-label text-[10px] text-on-surface-variant">{file}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Training Lifecycle */}
        <section className="mb-16">
          <h3 className="mb-10 font-headline text-2xl font-bold">Training Lifecycle</h3>
          <div className="relative">
            <div className="absolute left-0 top-1/2 hidden h-0.5 w-full -translate-y-1/2 bg-outline-variant md:block" />
            <div className="relative z-10 grid grid-cols-1 gap-8 md:grid-cols-4">
              {[
                ['1', 'Data Acquisition', 'Raw CAN bus telemetry is captured from production automotive networks via OBD-II interface'],
                ['2', 'Offline Pre-Training', 'DistilBERT fine-tuning via MLM on benign sequences, followed by Stage-2 classifier training'],
                ['3', 'Threshold Calibration', 'Anomaly decision boundary is established at the 99th percentile of the normal score distribution'],
                ['check', 'Deployment', 'Trained artifacts are serialized and served via the FastAPI inference backend'],
              ].map(([step, title, desc], index) => (
                <div key={title} className="flex flex-col items-center">
                  {index === 3 ? (
                    <div className="mb-4 flex h-8 w-8 items-center justify-center rounded-full bg-secondary-container text-on-secondary-container">
                      <span className="material-symbols-outlined text-lg">{step}</span>
                    </div>
                  ) : (
                    <div className="mb-4 flex h-8 w-8 items-center justify-center rounded-full bg-primary font-bold text-white">{step}</div>
                  )}
                  <div className={`w-full rounded-xl p-5 text-center ${index === 3 ? 'border border-secondary-container bg-surface-container-lowest' : 'bg-surface-container-lowest'}`}>
                    <h6 className="mb-1 text-sm font-bold">{title}</h6>
                    <p className="text-xs text-on-surface-variant">{desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Full Pipeline */}
        <section className="mb-16 rounded-2xl border border-outline-variant/20 bg-surface-container-low p-8 md:p-10">
          <div className="mb-8 flex items-center justify-between gap-4">
            <h3 className="font-headline text-2xl font-bold text-primary">End-to-End Inference Pipeline Specification</h3>
            <span className="rounded-full bg-secondary-fixed px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-secondary">
              Production Flow
            </span>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-8">
            {[
              ['1', 'Capture', 'Raw CAN frames are ingested from the live bus via OBD-II or SocketCAN interface'],
              ['2', 'Normalize', 'Hexadecimal arbitration IDs are canonicalized and padded to uniform length'],
              ['3', 'Window', '64-frame rolling sequence is constructed with single-frame stride'],
              ['4', 'Encode', 'DistilBERT contextual embedding and MLM reconstruction loss computation'],
              ['5', 'Gate', 'Binary threshold comparison: score > τ → ATTACK, else NORMAL'],
              ['6', 'Classify', 'Conditional Random Forest multiclass attack taxonomy decision'],
              ['7', 'Publish', 'Structured JSON payload emitted via REST endpoint and WebSocket channel'],
              ['8', 'Visualize', 'SOC dashboard renders anomaly scores, threat logs, and forensic analytics'],
            ].map(([step, title, desc]) => (
              <div key={title} className="rounded-xl border border-outline-variant/30 bg-surface-container-lowest p-4">
                <div className="mb-2 inline-flex h-6 min-w-6 items-center justify-center rounded-full bg-primary px-2 text-xs font-bold text-white">
                  {step}
                </div>
                <h4 className="mb-1 text-sm font-bold text-primary">{title}</h4>
                <p className="text-xs leading-relaxed text-on-surface-variant">{desc}</p>
              </div>
            ))}
          </div>

          <div className="mt-8 rounded-xl bg-surface-container-lowest p-4">
            <p className="font-mono text-xs text-on-surface-variant">
              Decision rule: score {'>'} threshold {'⇒'} ATTACK classification. If ATTACK, the Stage-2 Random Forest ensemble identifies the specific attack vector from the taxonomy: DoS, Fuzzy Injection, Gear Spoofing, or RPM Manipulation.
            </p>
          </div>
        </section>
      </main>
    </ConsoleLayout>
  );
}

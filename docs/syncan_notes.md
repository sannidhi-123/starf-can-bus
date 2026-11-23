# SynCAN Dataset Technical Notes & Data Verification

## 1. Overview & Provenance
* **Dataset Name:** SynCAN (Synthetic CAN Bus Benchmark Dataset)
* **Authors:** M. Hanselmann, T. Strauss, K. Dormann, H. Ulmer (ETAS GmbH & Robert Bosch GmbH)
* **Citation:** *CANet: An Unsupervised Intrusion Detection System for High Dimensional CAN Bus Data*, IEEE Access, 2020.
* **Purpose:** Multi-signal benchmark designed to evaluate CAN bus Intrusion Detection Systems on subtle payload and signal-space attacks, in addition to timing attacks.

---

## 2. Empirical Verification from Raw Data

### Time Column
* **Unit:** **Milliseconds (`ms`)** (floating-point).
* **Empirical Bus Characteristics:**
  * Mean bus inter-arrival time across all messages: **~2.09 ms** (min: 0.0018 ms, max: ~10.97 ms).
  * Per-ID transmission frequencies are strictly periodic multiples of 15 ms:
    * 15 ms cycle: `id1`, `id3`, `id5`, `id7`, `id8`
    * 30 ms cycle: `id2`, `id6`, `id9`
    * 45 ms cycle: `id4`, `id10`

### Identifier (`ID`) Values
* Exactly **10 discrete categorical identifiers**: `['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10']`.
* Unlike raw 11-bit or 29-bit hexadecimal arbitration IDs from physical vehicles (e.g., `0x18F`), SynCAN abstracts arbitration IDs into string tokens `id1` through `id10`.

### Signal Columns (`Signal1` – `Signal4`)
* **Semantic Meaning:** Extracted floating-point physical signal values representing simulated vehicle sensor/actuator signals (e.g., wheel speeds, steering angle, engine RPM).
* **Signal Counts per ID:** The number of valid signals varies per ID according to the tuple `(2, 3, 2, 1, 2, 2, 2, 1, 1, 4)`:
  * `id1`: 2 signals (`Signal1`, `Signal2`)
  * `id2`: 3 signals (`Signal1`, `Signal2`, `Signal3`)
  * `id3`: 2 signals (`Signal1`, `Signal2`)
  * `id4`: 1 signal (`Signal1`)
  * `id5`: 2 signals (`Signal1`, `Signal2`)
  * `id6`: 2 signals (`Signal1`, `Signal2`)
  * `id7`: 2 signals (`Signal1`, `Signal2`)
  * `id8`: 1 signal (`Signal1`)
  * `id9`: 1 signal (`Signal1`)
  * `id10`: 4 signals (`Signal1`, `Signal2`, `Signal3`, `Signal4`)
* **Schema Discrepancy:**
  * In `train_*.csv`: columns are named `Label,Time,ID,Signal1,Signal2,Signal3,Signal4`.
  * In `test_*.csv`: columns are named `Label,Time,ID,Signal1_of_ID,Signal2_of_ID,Signal3_of_ID,Signal4_of_ID`.
  * Signals not used by an ID are empty / `NaN` in raw CSV rows.

### Label Semantics & Attack Windows
* **Values:** `0` = Normal, `1` = Attack.
* **Training Sets (`train_1.csv` .. `train_4.csv`):** 100% `Label == 0` (unsupervised clean baseline).
* **Test Sets (`test_*.csv`):**
  * `test_normal.csv`: 100% `Label == 0` (clean evaluation trace).
  * `test_flooding.csv`, `test_plateau.csv`, `test_continuous.csv`, `test_playback.csv`, `test_suppress.csv`: contain interleaved periods of normal traffic and injected attack intervals.
  * **Critical Label Behavior:** During an active attack interval, **every message on the CAN bus is marked `Label = 1`**, regardless of which specific ID is being manipulated.

---

## 3. Attack Scenarios
1. **Flooding:** High-frequency insertion of existing CAN IDs.
2. **Plateau:** Target signal value is fixed/frozen to a constant value over time.
3. **Continuous:** Target signal drifts gradually away from authentic physical dynamics.
4. **Playback:** Historical legitimate segment of signal values is replayed.
5. **Suppress:** ECU transmission omission (message silence for an expected periodic ID).

---

## 4. License Terms Summary
* **Licensor:** ETAS GmbH and Robert Bosch GmbH.
* **Usage:** Freely available for non-commercial academic research, teaching, and simulation experiments.
* **Prohibitions:** Commercial deployment or field use in production vehicles is prohibited. Raw or modified dataset redistribution is restricted; derivative abstract works (e.g. trained models or feature summaries that do not allow reconstructing the original dataset) are permitted with citation.

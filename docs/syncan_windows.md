# SynCAN Window-Level Feature Engineering Documentation

## 1. Motivation & Context
In the SynCAN benchmark dataset, ground-truth attack labels are **window-tagged**: during an active attack interval, **every message on the CAN bus is tagged `Label = 1`**, regardless of whether it is an authentic sensor frame from a non-compromised ECU or a manipulated message. 

Consequently, a per-message classifier cannot cleanly isolate subtle in-bounds signal manipulation attacks (Plateau, Continuous Drift, Playback) on single messages without suffering massive false-positive rates on legitimate interleaved messages (message-level precision ~0.03, window false alarm rate ~100%).

To overcome this structural artifact of the dataset, STARF aggregates continuous CAN message streams into non-overlapping, fixed-duration time windows ($T = 0.5\text{ s}$).

---

## 2. Window Definition & Label Assignment
Let $\mathcal{M}_k = \{(t_i, \text{ID}_i, \mathbf{s}_i, y_i)\}_{i=1}^{M}$ represent the ordered sequence of CAN frames arriving in time window $W_k = [k \cdot T, (k+1) \cdot T)$.

### Window Ground-Truth Label Assignment (50% Rule)
$$\text{pct\_attack} = \frac{1}{|\mathcal{M}_k|} \sum_{m \in \mathcal{M}_k} \mathbb{I}(y_m = 1)$$

$$\text{Label}(W_k) = \begin{cases} 1, & \text{if } \text{pct\_attack} \ge 0.50 \\ 0, & \text{otherwise} \end{cases}$$

When $\text{Label}(W_k) = 1$, the window inherits the attack scenario label (e.g. `flooding`, `plateau`, `continuous`, `playback`, `suppress`). Otherwise, the window is classified as `normal`.

---

## 3. Mathematical Feature Definitions

### 3.1 Bus-Level Features (2 features)
1. **Total Message Count (`bus_total_count`):**
   $$N_{\text{bus}} = |\mathcal{M}_k|$$
   *Detection objective:* Drastically spikes during **Flooding** attacks.

2. **Largest Inter-Message Gap (`bus_max_gap`):**
   $$G_{\text{max}} = \max_{j=2}^{M} (t_j - t_{j-1})$$
   *Detection objective:* Anomalously large gaps indicate message silence or **Suppress** attacks.

---

### 3.2 Per-ID Features (22 features $\times$ 10 IDs = 220 features)
For each discrete CAN ID $c \in \{\text{id1}, \text{id2}, \dots, \text{id10}\}$, let $\mathcal{M}_{k, c} \subseteq \mathcal{M}_k$ denote the subset of frames matching ID $c$. Let $m_c = |\mathcal{M}_{k, c}|$. If $m_c = 0$, all features are set to $-1.0$ and the presence indicator is set to $0$.

#### Indicator & Counts (2 features)
3. **Presence Indicator (`id{c}_present`):**
   $$P_c = \mathbb{I}(m_c > 0)$$
   *Detection objective:* Drops to 0 during **Suppress** attacks targeting specific IDs.

4. **Message Count (`id{c}_count`):**
   $$C_c = m_c$$

#### Inter-Arrival Timing (2 features)
Let $\{t_1, t_2, \dots, t_{m_c}\}$ be the arrival timestamps for ID $c$ with $\Delta t_j = t_j - t_{j-1}$.
5. **Mean Inter-Arrival Time (`id{c}_dt_mean`):**
   $$\overline{\Delta t_c} = \frac{1}{m_c - 1} \sum_{j=2}^{m_c} \Delta t_j \quad (m_c \ge 2)$$

6. **Standard Deviation of Inter-Arrival Time (`id{c}_dt_std`):**
   $$\sigma_{\Delta t_c} = \sqrt{\frac{1}{m_c - 1} \sum_{j=2}^{m_c} (\Delta t_j - \overline{\Delta t_c})^2} \quad (m_c \ge 2)$$

#### Signal Distribution & Anomaly Metrics ($4 \text{ signals} \times 4 = 16$ features)
For valid signals $s \in \{1, 2, 3, 4\}$ of ID $c$, let $v_1, \dots, v_{m_c}$ be the observed values. Let $\mu_{c,s}$ and $\sigma_{c,s}$ denote the precomputed normal baseline mean and standard deviation from `models/syncan_stats.json`.

7. **Standardized $z$-Score Maximum (`id{c}_s{s}_max_abs_z`):**
   $$z_j = \frac{v_j - \mu_{c,s}}{\sigma_{c,s} + \epsilon}, \quad Z_{\text{max}} = \max_{j} |z_j|$$

8. **Standardized $z$-Score Mean (`id{c}_s{s}_mean_z`):**
   $$\overline{Z} = \frac{1}{m_c} \sum_{j=1}^{m_c} z_j$$

9. **Signal Standard Deviation Ratio (`id{c}_s{s}_std_ratio`):**
   $$\sigma_{\text{win}} = \sqrt{\frac{1}{m_c} \sum_{j=1}^{m_c} (v_j - \overline{v})^2}, \quad R_{\sigma} = \frac{\sigma_{\text{win}}}{\sigma_{c,s} + \epsilon}$$
   *Detection objective:* In normal operations, authentic sensor noise results in $R_{\sigma} \approx 1$. During a **Plateau** attack, the signal is frozen flat, driving $\sigma_{\text{win}} \to 0 \implies R_{\sigma} \to 0$.

10. **Fraction of Consecutive Identical Values (`id{c}_s{s}_frac_repeat`):**
    $$F_{\text{repeat}} = \frac{1}{m_c - 1} \sum_{j=2}^{m_c} \mathbb{I}(v_j = v_{j-1}) \quad (m_c \ge 2)$$
    *Detection objective:* Spikes towards $1.0$ under **Plateau** attacks.

#### Drift & Regression Slope (1 feature)
11. **Absolute Slope of Signal 1 (`id{c}_s1_slope`):**
    Using ordinary least squares regression over time within the window:
    $$\Delta \tau_j = t_j - \overline{t}, \quad \Delta v_j = v_j - \overline{v}$$
    $$\text{Slope} = \left| \frac{\sum_{j=1}^{m_c} \Delta \tau_j \Delta v_j}{\sum_{j=1}^{m_c} (\Delta \tau_j)^2} \right|$$
    *Detection objective:* Catches linear or polynomial signal deviation in **Continuous** drift attacks.

#### Sequence Correlation (1 feature)
12. **Previous Window Sequence Correlation (`id{c}_s1_prev_corr`):**
    Let $\mathbf{v}^{(k-1)}$ and $\mathbf{v}^{(k)}$ be the signal 1 series for ID $c$ in the previous window $W_{k-1}$ and current window $W_k$ respectively.
    $$L = \min(|\mathbf{v}^{(k-1)}|, |\mathbf{v}^{(k)}|)$$
    $$r = \frac{\sum_{j=1}^L (v_j^{(k-1)} - \overline{v^{(k-1)}})(v_j^{(k)} - \overline{v^{(k)}})}{\sqrt{\sum (v_j^{(k-1)} - \overline{v^{(k-1)}})^2 \sum (v_j^{(k)} - \overline{v^{(k)}})^2}}$$
    *Detection objective:* Identifies **Playback** attacks where replayed message sequences exhibit repetitive correlation across consecutive windows.

---

## 4. Total Dimensionality
$$\text{Total Features} = 2 \text{ (Bus)} + 10 \times 22 \text{ (Per-ID)} = 222 \text{ Features}$$
All features are non-overlapping, strictly causal, and NaN-safe.

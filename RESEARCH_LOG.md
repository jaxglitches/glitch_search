# Research log — searching LISA data for glitches in the WDM plane

Follow-up to the jaxglitches inference paper (`../jaxglitches_base_paper/`), which assumed
the glitch had already been found. This project supplies the search. Code in this folder,
paper draft in `../paper/main.tex`. Every number below comes from a script in `scripts/`
(CPU, deterministic); `README.md` has the quick start.

---

## 2026-09-22 — Session 1: an excess-power search

**Brief.** Search one year of equal-arm TDI-1 data (SciRD noise of the jaxglitches examples)
for glitches by excess power along the time axis of the WDM plane — not matched filtering —
starting from the frequency domain and cutting high frequencies.

### Design decisions, and the numbers behind them

- **Frequency cut: dt = 60 s (f_Nyq = 8.33 mHz), N = 2^19 (T_obs = 364.09 d).** The cut is set
  by where a glitch's *SNR* ends, not its power (`design_tiling.py`, band study):

  | fraction of ρ² below | 1 mHz | 3 mHz | 5 mHz | 8.33 mHz |
  |---|---|---|---|---|
  | τ = 0.1–1 s | 0.18 | 0.70 | 0.91 | 0.980 |
  | τ = 30 s | 0.25 | 0.84 | 0.98 | 0.9988 |
  | τ = 300 s | 0.88 | 0.998 | 1.000 | 1.000 |

  For τ → 0 the TDI-1 response is flat up to the knee; the OMS noise in fractional frequency
  rises as f⁴ above ~3 mHz, so the SNR density falls as f⁻⁴. The user's intuition ("glitches
  have little high-frequency content") holds in SNR terms, and the cause is the noise.
  The inference paper's 3 mHz cut would cost short glitches 30% of ρ².
- **A + E only.** T carries 0.04% (τ = 1e4 s) to 0.69% (τ = 0.1 s) of ρ².
- **Test-mass invariance.** Equal-arm relabellings act on (A, E) as rotations/reflections and
  S_A = S_E, so A + E pixel power is identical for all six test masses (tested pixel by pixel).
- **Whitening before the WDM transform**, then a kernel that reads the one-sided spectrum
  directly (one IFFT of length nt per interior channel). Pixels are exactly iid N(0, 1),
  Σw² = ρ². Agrees with `wdm_transform.WDM.from_time_series` to 5e-16. Post-transform
  whitening (the inference paper's σ²_nm = N S(f_m)/2δt) assumes S flat across a channel,
  which fails in the lowest channels.
- **Tiles**: k ∈ {1, 2} time bins × low-pass band [1, m_hi], m_hi on a √2 ladder to 6 mHz;
  energy exactly χ²(2 k m_hi). 114 shapes, 1.27e6 tiles/yr. Bands with lower edges never
  helped (ρ₅₀ +0.5–1.5% at every τ). Far tail by Lentz continued fraction (finite log p for
  ρ ~ 1e3).
- **Ladder of five tilings** nt = 512…131072 (Δt = 17 h … 4 min). Predicted ρ₅₀ at 1/yr:
  7.1–8.4 for τ = 0.1 s–2.3e4 s (9.9 at 5e4 s), ≤ 15% above the best single tiling for
  τ = 1–1e4 s. Minimax single tiling: nt = 8192 (64 min), ≤ 52% worse. All nine dyadic levels
  narrow the loss to 9%.
- **Events**: clusters per tiling, associated finest-first; a coarser cluster absorbs
  overlapping events only if its −log p ≥ 2× theirs (keeps long glitches whole, keeps pairs
  in one 17-h bin apart).

### False-alarm calibration

- Tile p-values uniform (1.005e-2, 9.81e-4, 9.88e-5 below 1e-2, 1e-3, 1e-4).
- **Pitfall found and fixed:** clustering is not nested in the threshold. Counting the events
  of one loose run above a stricter threshold undercounts by 25–35% (first calibration said
  1.00/yr; fresh noise gave 1.34/yr, the injection years 1.54/yr). Fix: score tiles once per
  noise year, rerun clustering at every threshold (`far.py`).
- 400 noise years: 1/yr at log₁₀ p = −5.915 (Bonferroni −6.104); n_eff = 0.74 n_trials.
  Validation on 300 fresh years: 1.07 ± 0.06/yr (target 1), 10.24 ± 0.19/yr (target 10).
  Robust (median-renormalised) mode: −5.927, discount 0.767.

### Sensitivity (2700 injections, `efficiency.py --per-cell 60`)

- Measured ρ₅₀ = 6.71, 7.12, 6.81, 6.91, 7.21 at τ = 1, 30, 300, 3000, 2e4 s (prediction
  7.1–8.3, conservative: it credits only the best tile). ≥ 92% found at ρ = 9, 100% at 12.
- Events matching no injection: 74 in 67.8 yr = 1.09/yr.
- Timing (ρ ≥ 10): median t̂ − t₀ = 19, 71, 365, 1070, 4770 s; t̂ is the excess-power
  centroid and lags the onset by O(τ). ρ_ep/ρ = 0.71–0.82.

### Realistic years (`run_search.py`)

- **Showcase** (9 glitches, τ = 0.5 s–3e4 s, ρ = 6–100, incl. the fiducial t₀ = 400 s,
  τ = 300 s, ρ = 42.7 straddling the year's join): all found, no false alarm; fiducial at
  −log₁₀ p = 344 on all five tilings.
- **LPF-like year** (seed 1): 1597 glitches (10 bootstrap draws of the two inconsistent rows
  dropped), ρ > 8: 502, ρ > 20: 159, ρ > 100: 6, max 529. 619 events: 612 glitch, 6 assisted,
  0 pile-up, 1 noise (1.0 expected); none split. Found fraction tracks isolated injections.
  Loud misses (ρ > 12): 6 this year; over five years 36 of 2014 (1.8%), 32 of them with a
  glitch ≥ half as loud within 30 min → unresolved pairs. *(Corrected in Session 2: this
  line first said 38 and 36. Rerunning `run_search.py lpf --seed 1..5` with the current code
  gives 6+8+7+4+11 = 36 misses and 6+6+7+2+11 = 32 pairs, identical to pass 0 of the loop.)*
- **Attribution by ground truth** (`truth.py`): time-window matching misattributes at the LPF
  rate (a glitch usually has a neighbour within an hour). Attribution uses the noiseless
  energy in each event's seed tile instead.
- **Inconsistent catalogue rows kept** (`--keep-all`): the 2.1e-8 m/s row reaches ρ ≈ 3e5; its
  wavelet sidelobes make 41 pile-up events and hide 14 extra loud glitches (20 loud misses).
- Cost: 0.19 s per year-long scan (CPU, after compilation); clustering 0.6 s for an LPF year.

---

## 2026-09-22 — Session 2: jaxmbhb, a matched filter, the follow-up, the paper

**Brief.** (1) Draft a paper from the code. (2) Take inspiration from the search in jaxmbhb.
(3) Keep this log.

### What the jaxmbhb search does (surveyed 2026-09-22)

The local clone (`~/Physics/LISA_repos/jaxmbhb/mbhb_fastlike`, stale since July) has no
search on `main`. Read-only `git ls-remote` + a blobless clone of
`gitlab.in2p3.fr/lisa-apc/mbhb_fastlike` and the lisa-apc group (in the session scratchpad,
not in the user's clone) show:

- **`mbhbh_search` branch** (S. Babak, 2026-06-23): log-likelihood maximised over a positive
  amplitude (Â = max(⟨d|h⟩/⟨h|h⟩, 0), LLR = ½⟨d|h⟩²/⟨h|h⟩) and over a circular time shift,
  all shifts at once by one inverse FFT of C_k = d_kᴴ S_k⁻¹ h_k, restricted to a prior window.
- **`time_freq_ll` branch** (J. Perret, 2026-06-26): `TFlikelihood` — full whitening
  u = Λ^{-1/2} V† D^{-1/2} d̃ (channel decorrelation, truncated pseudo-inverse) *before* the
  WDM transform, so the TF inner product equals the FD one to machine precision. Same choice
  as our whitening-first kernel (ours is diagonal because equal-arm AET is uncorrelated).
- **`MBHBSeeker`** (`jaxmbhb/search.py`, restored 2026-07-07, then moved out) and the notebook
  `tf_mbhb_search.ipynb` in **lisa-apc/geemoo-mbhb-detection** (J. Perret, 2026-07-17):
  - Stage A, model-free: whitened WDM power P = Σ_{A,E} W² (one tiling, ΔT = 3 h, dt = 5 s);
    per-pixel Q = −log₁₀ Q_Γ(k, P/s), s = median(P)/k; hysteresis clustering (seed Q ≥ 7,
    grow Q ≥ 2, 8-connected, ≥ 3 px); Newtonian f^{-8/3} ridge fit → (Mc, tc).
    (Earlier version: mean excess power along the chirp track on an (Mc, tc) grid.)
  - Stage B, matched filter: F = ½ max_{|τ|≤T} ⟨d|h_τ⟩²/⟨h|h⟩, amplitude analytic, tc by
    FFT; pre-localise tc once with a wide window, then a narrow window (±1500 s) so the time
    slide cannot absorb a wrong chirp mass; differential evolution over 9 parameters; the
    production likelihood then refines tc; seed quality by whitened overlap (≳ 0.97 good).
  - Roadmap: detect → subtract → re-detect.
  - Also in that repo: a port of the old Sangria F-statistic search (`premerger_detection`).
- Not public as a paper; cite as software, and check with Perret/Babak before a public version.

### What we took from it

1. **A matched filter for glitches**, built exactly like their Stage B, as a *yardstick*
   (the README's "ideal matched filter" was only a Rice-formula estimate).
2. **A follow-up** turning each excess-power event into a seed for jaxglitches.
3. **Subtract and search again**, their roadmap item, for the two failure modes of Session 1
   (unresolved pairs; very loud glitches whose sidelobes swamp their neighbours).

Not taken: their per-pixel threshold without FAR calibration (fine at MBHB SNRs of hundreds,
not for a glitch population that lives near threshold); DE (a glitch has one continuous
parameter left after the analytic maximisations).

### The glitch matched filter (`glitchsearch/mfilter.py`)

- h = Δv u(f; τ, ℓ) e^{-2πif t₀} exactly (checked in `jaxglitches.data`): Δv̂ = N/M,
  2 ln Λ = N²/M with N(t₀) = (d|u_{t₀}), z = N/√M ~ N(0, 1) under noise, = ρ at truth.
- **Test masses collapse to arms**: in A + E, 12 ≈ 21, 23 ≈ 32, 31 ≈ 13 (match > 0.9999 at
  every τ); arms are 60° apart (match 0.50). Bank uses one link per arm. The test mass is
  only measurable at ρ ≳ 100 — seeds should report the arm.
- **Decay times**: everything below ~35 s is one template (match > 0.99 with τ = 0.01 s);
  above ~100 s the family is self-similar in log τ (neighbour match → 0.9696 per 1/8 decade).
  Greedy bank at min match 0.98 over [0.01, 1e5] s: **17 decay times, 51 templates**
  (cached in `results/mf_bank.npz`, 25 s to build).
- **Onset grid**: at dt = 60 s a short glitch's correlation drops to 0.865 half a sample off
  peak; oversampling by 2 (30 s) → worst 0.965, mean loss ~1%. Scan cost: 0.9 s/yr (51
  irffts of 2^20).
- Noise check: z std 0.999–1.000 per template (0.984 for the 1e5 s template); best of 51
  over a noise year ≈ 5.4.
- **Calibration** (`calibrate_far.py --mf --years 400`, 629 s): 1/yr at |z| = 5.259
  (log₁₀ p = −6.840); 0.1/yr at 5.757; 10/yr at 4.810; n_eff = 0.22 × (templates × onsets).

### The follow-up (`mfilter.followup`) and the loop (`glitchsearch/pipeline.py`)

- Per event: best bank template with onset in the event window (core clusters ± max(one
  seed bin, 30 min)), then L-BFGS-B on (t₀, log τ) with jax gradients (data heterodyned to
  the bank onset first), for both test masses of the arm; Laplace widths from the Hessian.
  Noiseless check: recovers t₀, τ, Δv, ρ exactly (τ = 1 s: t₀ off by 2(τ − 0.01 s), the
  t₀ + 2τ degeneracy).
- **Bug found and fixed:** `event_window` took min/max over core clusters stored on
  different sides of the periodic year's join → a window spanning the whole year for the
  fiducial glitch (t₀ = 400 s), which then got the seed of the ρ = 100 glitch 44 d away.
  Fix: move each cluster by a whole period to lie nearest the seed cluster.
- **Showcase seeds** (`followup.py showcase`): all nine within 57 s of t₀; arm right 9/9,
  test mass 6/9 (degenerate, as expected). **Fiducial glitch: t₀ − 2.6 s (Laplace σ 6.7 s;
  the inference paper's posterior width was 7.0–7.1 s), τ = 297 s, Δv̂/Δv = 1.010, test mass
  12, ρ = 43.6.** The inference paper's heterodyne needs |t₀ − t_ref| ≲ 2.6e3 s: met by a
  factor ~400 (and by the excess-power centroid alone, 485 s).
- Loop: loud-first (events with ρ_ep > 1e3, one at a time), then pass 0 = the ordinary
  search, follow up + subtract everything, then passes ≥ 1 on the residual restricted to the
  previous pass's windows (extra FAR = FAR × window coverage, reported).

### Excess power against the matched filter (`efficiency.py --per-cell 60 --mf`, 19 min)

Same noise, same 2700 injections; the excess-power columns reproduce the run without `--mf`
draw for draw (ρ₅₀ 6.71, 7.12, 6.81, 6.91, 7.21; 74 unmatched events = 1.09/yr).

| τ [s] | 1 | 30 | 300 | 3000 | 2e4 |
|---|---|---|---|---|---|
| ρ₅₀ excess power | 6.71 ± 0.15 | 7.12 ± 0.16 | 6.81 ± 0.18 | 6.91 ± 0.16 | 7.21 ± 0.21 |
| ρ₅₀ matched filter | 5.06 ± 0.16 | 5.21 ± 0.14 | 5.20 ± 0.19 | 5.18 ± 0.15 | 5.00 ± 0.15 |
| ratio | 1.33 | 1.37 | 1.31 | 1.33 | 1.44 |

- The Rice estimate (4.6–5.4) was close. **Not assuming the waveform costs a factor
  1.3–1.4 in SNR.** Applied to the LPF-like year (seed 1), the two efficiency curves predict
  638 glitches found by excess power (618 actually) and 810 by the matched filter: +27%,
  all between ρ ≈ 5 and 7.
- Matched-filter events matching no injection: 91 in 67.8 yr = 1.34/yr (target 1). Not
  noise: of 27 in the first 25 years, 7 lie 2–13 h *before* an injection with τ ≥ 3e3 s,
  on a template with τ ≥ 2.6e3 s whose onset slid ahead of the glitch (outside the matching
  window, which opens 1 h before t₀); the other 20 → 0.80 ± 0.18/yr. A production matched
  filter should cluster by template duration, not a fixed 1-h gap.
- **Seeds** (follow-up of the excess-power events, ρ ≥ 10): median |t₀ error| 4.8, 14.9,
  15.8, 54, 140 s at τ = 1, 30, 300, 3000, 2e4 s (90%: 20, 53, 43, 124, 426 s) — all inside
  the 2.6e3 s heterodyne tolerance; the excess-power centroid alone had medians 19–4770 s.
  Δv̂/Δv median 0.96–1.01; τ within 1% for τ ≥ 300 s (0.92 at 30 s, knee 5.3 mHz; τ → the
  0.01 s bank edge at 1 s, unresolved); ρ_seed/ρ 0.996–1.013; **arm right 100%**, test mass
  right 47–54% (the 12/21 degeneracy, as predicted).

### Speed of the refinement

The first LPF-year loop took > 76 min before being killed. One profile evaluation computed
the whole jaxglitches template (transcendentals over 2.6e5 bins) plus XLA thread
oversubscription with parallel jobs (45–80 ms per call). The decay time enters only as
q = 1/(2πfτ − i)², so V = U/q is computed once per test mass and an evaluation is one complex
exp per bin: identical results to all printed digits, 2–6× faster (≈ 0.1 s per refinement).
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"` lets several
jobs share the 22 cores at no cost per job.

### Bug: the loop subtracted loud glitches twice

First loop run on LPF year 1: passes [619, 46, 28], 14 new glitches, but 58 "pile-up" events
in passes 1–2. Diagnosis (per-event dump): a faint event within ~30 min of a loud one had a
padded window (≥ 30 min) reaching the loud glitch, so both events were seeded with it; it was
subtracted twice, its negative appeared in the residual as an event, which was seeded and
subtracted, and so on. Fix: `mfilter.partition_windows` clips each window at the midpoint
between neighbouring events' cores; `mfilter.dedupe` drops a seed that repeats a louder one of
the same pass (same arm, |Δt₀| ≤ max(120 s, τ/2)), its event being searched again next pass.
Regression test: a faint glitch 25 min before a loud one (the old windows seed both events
with the loud glitch at 25–30 min separations; at 20 min they merge into one event).
After the partition alone: ghosts 58 → 3 (the 3 from a triple within 25 min), new glitches
14 → 15; dedupe addresses the triple.

### Confirmation: subtract only what the matched filter believes

The one noise event of LPF year 1 was re-found in passes 1 and 2: its seed had ρ = 3.6, so
subtracting it changed nothing and its window kept producing the same excess. Measured
(`followup.py noise --years 40 --far 10`): 429 noise events have seed ρ with median 4.17,
95% 5.04, 99% 5.53, max 5.78 — 5.6% reach 5. Every pass-0 glitch/assisted event of LPF year 1
has ρ_seed ≥ 5.34 (1st percentile 6.25). Rule adopted: subtract, and re-search the windows
of, only seeds with ρ_seed ≥ 5 (`pipeline.run(rho_confirm=5)`).

With 160 noise years (`followup.py noise --years 160 --far 10`): 1564 events (9.8/yr, a
check of the 10/yr calibration), seed ρ median 4.15, 95% 5.03, 99% 5.43, max 5.97; 5.5%
confirmed at ρ_seed ≥ 5, 0.07/yr at ≥ 5.5, none at ≥ 6.

### A two-stage search (the jaxmbhb design, measured for glitches)

Trigger excess power at 10 false alarms/yr, keep events with ρ_seed ≥ 5
(`efficiency.py --per-cell 60 --mf --far 10`, same 2700 injections):

| τ [s] | 1 | 30 | 300 | 3000 | 2e4 |
|---|---|---|---|---|---|
| excess power @ 1/yr | 6.71 | 7.12 | 6.81 | 6.91 | 7.21 |
| excess power @ 10/yr | 6.17 | 6.06 | 6.11 | 6.30 | 6.47 |
| **@ 10/yr + ρ_seed ≥ 5 (0.54 ± 0.06 false alarms/yr)** | **6.17** | **6.06** | **6.15** | **6.33** | **6.65** |
| matched filter @ 1/yr | 5.06 | 5.21 | 5.20 | 5.18 | 5.00 |

9–15% less SNR than excess power alone, at half the false-alarm rate; within 1.16–1.33 of
the matched filter (was 1.31–1.44). Confirmation drops 1 of ~1470 injections found at ρ ≥ 8.
Unmatched excess-power events at 10/yr: 644 in 67.8 yr = 9.50/yr. The price: a transient the
matched filter does not recognise is vetoed, so the confirmed list is not model-independent
(keep both lists).

### The final loop

- LPF-like year 1: passes of 619, 16, 1 events; found 618 → 633 (15 glitches the first pass
  had merged, all 6 loud misses among them); 2 residual (pile-up) events in passes 1–2; the
  restricted passes add 0.095 false alarms/yr; 425 s.
- Keep-all year 1: the bootstrap draws the 2.1e-8 m/s row three times (ρ = 2.67e5, τ = 65 s).
  The loud-first stage recovers them with t₀ to 2 ms and Δv to 4e-6, and the year then runs
  exactly like the filtered one (619, 16, 1). The 41 sidelobe events vanish; ρ > 12: 409
  glitches, 9 missed after pass 0 (6 pairs + the 3 loud ones), 0 after the loop.
- Seeds of the population (keep-all year 1): τ < 53 s (624): median |t₀ err| 9.0 s, 90%
  35.9 s, 99% 68.5 s, none beyond 2.6e3 s; Δv̂/Δv 16–84% 0.92–1.13; arm 99.7%, test mass
  55%. τ > 53 s (11): median 1.7 s; Δv 0.97–1.03; τ ±0.02 dex; arm 100%.
- **Five LPF-like years** (`followup.py lpf --seeds 1 2 3 4 5`, 1959 s): passes [619, 16, 1],
  [644, 14, 1], [649, 20], [591, 13, 2], [586, 19]; found 3072 → 3152 (+80); ρ > 12: 2014
  glitches, 36 missed after pass 0 (32 pairs) → 4 after the loop (3 pairs); by kind — pass 0:
  3019 glitch, 54 assisted, 8 pile-up, 8 noise; pass 1: 74 glitch, 4 assisted, 4 pile-up;
  pass 2: 1 glitch, 1 assisted, 2 pile-up (dedupe dropped 3 seeds in all); extra FAR
  0.097/yr. Seeds: τ < 53 s (3111): median |t₀ err| 9.7 s, 90% 35.4 s, 99% 65.4 s, 1 beyond
  2.6e3 s; Δv̂/Δv 0.917–1.129 (16–84%); arm 99.8%, test mass 54.5%. τ > 53 s (39): median
  2.9 s, 99% 8.0 s; Δv 0.961–1.038; τ −0.018/+0.021 dex; arm 100%.

### Paper

`../paper/main.tex` (+ `refs.bib`, `figures/` copied from `figures/` here): a complete draft,
15 pages, compiles cleanly with `latexmk -pdf` (no undefined references, no overfull boxes),
every number from the runs above. Structure: introduction; data and the band cut; whitened
WDM pixels (incl. the 25–27% error of post-transform whitening in the lowest channels);
tiles and the ladder; the false-alarm rate; sensitivity against the matched filter; from
events to seeds (the fiducial glitch end to end); a realistic year (showcase, LPF-like year,
the loop, the loud tail); discussion (the MBHB search compared, the two-stage search,
assumptions, cost); conclusions; appendices (χ² tail, reproducing the numbers).
To do before circulating: confirm the [CHECK] references in `refs.bib` (volume/page, all
from memory); the jaxmbhb/geemoo work is unpublished software, so ask J. Perret and S. Babak
how they want it cited; the acknowledgements are a placeholder.

### Open questions / next steps

- Real-data edges (red-noise jump at the record's ends) — mask/pad or edge-aware noise model.
- Glitch vs GW classification (MBHBs are transients too): band top, seed tiling, (A, E)
  direction ∈ 3 arms, follow-up residual.
- Unequal arms / TDI-2: whitening with channel decorrelation (as in jaxmbhb `time_freq_ll`).
- Hand the seeds to jaxglitches and compare the posterior with the truth-started one of the
  inference paper — the experiment its conclusions call for.
- Shapes beyond n = 1: excess power is template-agnostic, the follow-up is not.
- Two-stage search: tune trigger FAR and ρ_seed jointly (≥ 5.5 gives 0.07/yr at 10/yr
  triggers); measure a looser trigger (100/yr).
- Pairs: the loop's seeds are fitted in sequence; a joint re-fit of overlapping seeds
  (backfitting) or RJ would improve the first seed of a pair.
- The matched filter as a search clusters with a fixed 1-h gap, which splits long glitches
  (1.34 unmatched/yr); cluster by template duration instead.

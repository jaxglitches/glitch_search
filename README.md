# glitchsearch — an excess-power search for LISA glitches in the WDM plane

A draft of the search stage that precedes the glitch inference of `jaxglitches`. One year
of equal-arm TDI-1 data (the jaxglitches templates, and the SciRD noise of the jaxglitches
examples) is simulated in the frequency domain, whitened, transformed to a ladder of
Wilson–Daubechies–Meyer tilings straight from the one-sided spectrum, and scanned along the
time axis for excess power. The search itself does no matched filtering: every candidate
gets an exact χ² p-value and a false-alarm rate measured on noise. A matched filter built
like the MBHB search of jaxmbhb (Δv analytic, t₀ by FFT) then serves twice: as the
yardstick for what not assuming the waveform costs, and as a follow-up that turns each
event into a seed (t₀, Δv, τ, test mass) for jaxglitches — after which the seeds are
subtracted and the residual searched again.

The paper draft is `../paper/main.tex`; `RESEARCH_LOG.md` records what was done, found and
fixed, session by session.

## Quick start

Everything runs in the jaxglitches environment (jax 0.10, `wdm_transform`, matplotlib):

```sh
PY=/home/giorgio/Desktop/jaxglitches/.venv/bin/python
cd /home/giorgio/Desktop/glitch_search/code

$PY -m pytest -q tests                      # 24 tests, ~15 s
$PY scripts/design_tiling.py --banks        # why dt = 60 s, A+E, and the WDM ladder   (~3 min)
$PY scripts/calibrate_far.py --years 400    # false-alarm rate measured on noise        (~2 min)
$PY scripts/efficiency.py --per-cell 60     # 2700 injections vs the analytic prediction (~1 min)
$PY scripts/run_search.py showcase          # one year, nine glitches, figures          (seconds)
$PY scripts/run_search.py lpf               # one year of LPF-like glitches             (~30 s)

$PY scripts/calibrate_far.py --mf --years 400      # the matched filter's FAR          (~10 min)
$PY scripts/efficiency.py --per-cell 60 --mf       # both searches, same injections, seeds (~20 min)
$PY scripts/followup.py showcase                   # seeds of the nine glitches        (~20 s)
$PY scripts/followup.py lpf --seeds 1 2 3 4 5      # the search-subtract loop, 5 years (~35 min)
$PY scripts/followup.py lpf --keep-all             # ... with the rho ~ 3e5 catalogue row (~7 min)
$PY scripts/followup.py noise --years 160 --far 10 # seed SNR of noise events          (~20 min)
$PY scripts/efficiency.py --per-cell 60 --mf --far 10   # the two-stage search         (~20 min)
```

For several jobs at once, `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false
intra_op_parallelism_threads=1"` keeps each one from claiming every core; it does not slow
them down.

Every script writes `results/*.npz` and `figures/*.pdf`, and `design_tiling.py` and
`efficiency.py` take `--figure` to redraw from the stored `.npz`. `calibrate_far.py`,
`efficiency.py` and `run_search.py` take `--robust` for the median-renormalised mode. The
scripts default to the CPU (`JAX_PLATFORMS=cpu`, overridable): nothing here needs a GPU, and
the CPU keeps the printed numbers identical from run to run.

```python
import jax.random as jr
from glitchsearch import Grid, Search, noise_fd, glitch_fd, with_snr, load_calibration

grid = Grid()                                            # 1 yr, dt = 60 s, N = 2^19
search = Search(grid, calibration=load_calibration("results"))
p = with_snr([100 * 86400.0, 1e-12, 300.0], 12.0, grid)[0]   # [t0, Deltav, tau] at rho = 12
events, scans = search.run(noise_fd(jr.PRNGKey(0), grid) + glitch_fd(p, grid), far_per_year=1.0)
events[0].summary()   # t_hat, log10p, log10_far, rho_ep, seed tiling, band top f_hi, levels
```

## Design, and the numbers behind it

**Cutting high frequencies (dt = 60 s, f_Nyq = 8.33 mHz).** Your intuition holds in terms of
SNR. Below 8.33 mHz every glitch keeps ≥ 98.0% of its ρ² (A+E+T, over 10⁻⁵–1 Hz), and
anything with τ ≳ 30 s keeps > 99.9%. For the shortest glitches the cause is the noise, not
the glitch: for τ → 0 the TDI-1 response is flat up to the knee, but the OMS noise in
fractional frequency rises as f⁴ above ~3 mHz. The 3 mHz band of the inference paper would
cost those glitches 30% of ρ². Working in the frequency domain means the record never
exists above 8.33 mHz: 2.6×10⁵ bins per channel instead of 1.3×10⁸ samples at 4 Hz. The
best search tile for any glitch tops out at 4–5 mHz in any case. N = 2¹⁹ (364.1 d) is a
power of two, so every dyadic tiling divides the record exactly.

**A + E only.** T carries < 0.7% of a test-mass glitch's ρ², at every τ. The A + E pixel
power is also exactly invariant under the choice of test mass (the six relabellings act on
(A, E) as rotations and reflections, and S_A = S_E), which `tests/test_search.py` checks
pixel by pixel.

**Whitening before the transform.** The spectrum is divided by the per-bin noise σ_k (the
jaxglitches bin variance) *before* the WDM transform. The noise is then white, and the WDM
basis is orthonormal, so pixels are exactly iid N(0, 1) however coloured S(f) is, while a
signal satisfies Σ w² = ρ². Whitening pixel by pixel afterwards would assume S flat across
each channel, which fails in the lowest channels, where S goes as f⁻². The kernel
(`glitchsearch/wdm.py`) is `wdm_transform`'s restricted to interior channels and fed the
one-sided spectrum. It agrees with `WDM.from_time_series` to 5×10⁻¹⁶. Channel m only reads
bins below (m+1)·nt/2, so the frequency cut and the top channel are the same statement.

**The number of time bins.** The best excess-power tile tracks the glitch's whitened
duration, which runs from minutes (τ → 0, set by the noise bandwidth) to days (τ ~ 10⁴ s).
The table gives the SNR needed for 50% detection at 1 false alarm/yr (Bonferroni, median
over placements; `design_tiling.py`):

| τ [s] | 1 | 10 | 100 | 10³ | 10⁴ | 5×10⁴ |
|---|---|---|---|---|---|---|
| nt = 131072 (Δt = 4 min) | 6.9 | 6.9 | 8.5 | 54 | 227 | 501 |
| nt = 32768 (16 min) | 8.2 | 8.1 | 7.2 | 9.9 | 35 | 80 |
| nt = 8192 (64 min) | 10.4 | 10.3 | 8.9 | 7.1 | 12.1 | 26.7 |
| nt = 2048 (4.3 h) | 13.7 | 13.6 | 11.5 | 8.5 | 7.0 | 12.9 |
| nt = 512 (17 h) | 18.4 | 18.2 | 15.2 | 10.9 | 7.8 | 8.5 |
| **ladder of all five (default)** | **7.1** | **7.1** | **7.4** | **7.5** | **7.8** | **9.9** |
| ideal matched filter (Rice estimate, 10 τ templates) | 5.4 | 5.4 | 5.2 | 5.0 | 4.7 | 4.5 |

Measured on 2700 injections at the calibrated 1/yr thresholds (`efficiency.py --mf`), at
τ = 1, 30, 300, 3000, 2×10⁴ s: excess power ρ₅₀ = 6.71, 7.12, 6.81, 6.91, 7.21; the matched
filter of `mfilter.py` 5.06, 5.21, 5.20, 5.18, 5.00 (± 0.15).

No single tiling works across the population, so the search runs a factor-4 ladder
(`grid.LADDER`). It never costs more than 15% against the best single tiling at any τ in
1–10⁴ s. It scans 2.4 times the tiles of its finest level alone, which raises the threshold
by 0.4 dex, a 3% cost in SNR (6.9 → 7.1 for short glitches). If you want one grid, the
minimax choice over τ = 1–10⁴ s is nt = 8192 (Δt = 64 min, ΔF = 130 μHz; `grid.SINGLE`): at worst
52% above the best tiling, at τ → 0. Assuming only compactness, and no waveform, costs a
factor 1.31–1.44 in SNR against a matched filter calibrated the same way.

**The statistic.** Each tile is k ∈ {1, 2} consecutive time bins times the low-pass band of
channels 1..m_hi, with m_hi on a √2 ladder up to 6 mHz: 114 shapes, 1.27×10⁶ tiles per year.
Under noise its energy is χ² with 2·k·m_hi degrees of freedom. Bands with lower edges
m_lo > 1 were tried and never helped, at any τ (`design_tiling.py --banks`). Tails beyond
p ~ 10⁻³⁰⁰ come from a continued fraction, so a ρ = 100 glitch still gets a finite
log₁₀ p (−1528 in the showcase).

**Events.** Significant tiles are merged into clusters per tiling, and clusters are
associated across tilings from the finest down. A coarser cluster that overlaps an event
becomes its support; it absorbs the events it overlaps only if its −log p is at least twice
theirs. That rule keeps a long glitch, fragmented at fine resolution, as one event, and
keeps apart two short glitches that share a 17-h bin, which happens routinely at ~1.3
detectable glitches per day.

**The false-alarm rate is measured, not bounded.** Over 400 noise years (`calibrate_far.py`),
single-tile p-values are uniform (1.005×10⁻², 9.81×10⁻⁴, 9.88×10⁻⁵ below 10⁻², 10⁻³, 10⁻⁴).
A rate of 1/yr sits at log₁₀ p = −5.915, against −6.104 for the Bonferroni bound; 74% of the
tiles count as independent. On 300 fresh noise years the calibrated threshold gives
1.07 ± 0.06/yr for a target of 1/yr, and 10.2 ± 0.2/yr for a target of 10/yr. One pitfall
is worth knowing: clustering is not nested in the threshold. Thresholding the events of a
single loose run undercounts the rate at the operating point by ~25%, so the calibration
reruns the clustering at every threshold of a grid (`glitchsearch/far.py`).

## What it achieves

- **Efficiency** (`efficiency.py`, 2700 isolated injections at FAR = 1/yr): measured ρ₅₀ =
  6.7, 7.1, 6.8, 6.9 and 7.2 at τ = 1, 30, 300, 3000 and 2×10⁴ s, with ≥ 92% found by
  ρ = 9 and 100% by ρ = 12. The analytic prediction (`glitchsearch/design.py`) is 7.1–8.3.
  It is conservative, because it credits only the single best tile. Events with no
  injection: 1.09/yr.
- **Timing.** For glitches found at ρ ≥ 10, t̂ − t₀ has a median of 19 s (τ = 1 s), 71 s
  (30 s), 365 s (300 s), 1.1×10³ s (3000 s) and 4.8×10³ s (2×10⁴ s). t̂ is the centroid of
  the whitened excess power, so it lags the onset by O(τ); at ρ ≥ 8 the 5–95% spread is
  ±150 s for short glitches and hours for the longest. That is the t₀ prior to hand to parameter
  estimation. The paper's heterodyned grid needs t₀ to ~2.6×10³ s.
- **Showcase year** (`run_search.py showcase`): nine glitches, τ = 0.5 s–3×10⁴ s,
  ρ = 6–100, are all found with no false alarm. The paper's fiducial glitch (t₀ = 400 s,
  straddling the periodic year's join) comes out at −log₁₀ p = 344 on all five tilings. The
  seed tiling and band top track τ: 16 min / 2.3 mHz at τ = 300 s, 64 min / 0.46 mHz at
  3000 s, 17 h / 0.17 mHz at 3×10⁴ s.
- **LPF-like year** (`run_search.py lpf`: 1597 glitches at ~4.4/day, LPF catalogue
  bootstrapped). Of 619 events, 612 are made by one glitch, 6 by a sub-threshold glitch
  carried over threshold by noise, 0 by pile-up of faint glitches, and 1 by noise (1.0
  expected). No glitch is split into two events. The found fraction tracks the isolated
  injections, and events are attributed by the noiseless energy in their seed tile
  (`glitchsearch/truth.py`); time windows misattribute at this glitch rate. The loud
  misses are 6 of 406 glitches with ρ > 12 in this year, and 36 of 2014 over five
  simulated years (1.8%). 32 of those 36 have a glitch at least half as loud within 30 min,
  a few minutes away in this year's six, so they are unresolved pairs (the loop below
  recovers all but 4).
- **Unknown PSD** (`--robust`). Renormalising each WDM channel by its median power over the
  year leaves the showcase results unchanged (log p within a few per cent), and the
  calibrated 1/yr threshold moves from −5.915 to −5.927. Stationary lines, such as Galactic
  binaries, and a mis-estimated noise level are absorbed the same way.
- **Cost.** One year-long scan (five tilings, every tile) takes 0.19 s on the CPU after
  compilation. Clustering the ~600 events of an LPF-like year takes 0.6 s more.

## Matched filter, follow-up, and the loop (after jaxmbhb)

The MBHB search of jaxmbhb (lisa-apc: `mbhb_fastlike` branches `mbhbh_search` and
`time_freq_ll`; `MBHBSeeker` and `tf_mbhb_search.ipynb`, now in `geemoo-mbhb-detection`) runs
excess power in the whitened WDM plane first, then a matched filter maximised analytically
over the amplitude and by FFT over time, and hands the result to the sampler as a seed; its
roadmap ends with detect → subtract → re-detect. `mfilter.py` and `pipeline.py` do the same
for glitches (`RESEARCH_LOG.md` has the survey).

- **The matched filter.** h = Δv u(f; τ, test mass) e^{-2πift₀} exactly, so Δv̂ = N/M and
  every t₀ of the year is one inverse FFT. The six test masses collapse to three arms in
  A + E (12 ≈ 21 etc. to < 1e-4 mismatch), every τ below ~35 s is one template, and a bank at
  minimum match 0.98 over τ = 0.01–1e5 s has 17 decay times: 51 templates, t₀ on a dt/2 grid.
  Calibrated like the excess-power search (400 noise years): 1/yr at |z| = 5.26. Its events
  matching no injection run at 1.34/yr; the excess is 7 events (per 25 yr) 2–13 h *before*
  long injections, on long templates whose onset slid ahead — not noise (the rest: 0.80/yr).
- **The follow-up** (`mfilter.followup`): the best bank template in each event's window, then
  L-BFGS on (t₀, log τ) with jax gradients for both test masses of the arm (τ enters only as
  (2πfτ − i)⁻², so an evaluation is one complex exp per bin, ~0.1 s per refinement). Seeds of
  isolated glitches found at ρ ≥ 10: median |t₀ error| 5, 15, 16, 54, 140 s at τ = 1, 30, 300,
  3000, 2e4 s (90%: ≤ 426 s, always inside the 2.6e3 s heterodyne tolerance of the inference
  paper); Δv̂ within 4%, τ within 1% for τ ≥ 300 s, arm always right, test mass 50/50.
  The fiducial glitch of the paper: t₀ − 2.6 s (Laplace σ 6.7 s), τ = 297 s, Δv̂/Δv = 1.010.
- **Windows are partitioned and seeds de-duplicated.** A faint event within ~30 min of a loud
  one once reached the loud glitch with its padded window; both events got its seed, it was
  subtracted twice, and its negative came back as 58 ghost events in one LPF-like year.
- **The loop** (`pipeline.run`): loud glitches first (ρ_ep > 1e3, one at a time); then pass 0
  = the ordinary search, every event followed up; seeds with ρ_seed ≥ 5 (confirmed) are
  subtracted and only their windows searched again, until a pass adds nothing. Noise events
  have ρ_seed median 4.2, 95% < 5.0 (429 events in 40 noise years at 10/yr); every glitch
  event of the LPF-like year has ρ_seed ≥ 5.3. LPF-like year 1: passes of 619, 16 and 1
  events; 15 glitches the first pass had merged are found (all 6 loud misses), 2 residual
  events, +0.095 false alarms/yr. **Five LPF-like years**: found 3072 → 3152; of 2014
  glitches with ρ > 12, 36 missed by the single pass (32 pairs) → 4 after the loop; 6
  residual events in the later passes in five years; +0.097 false alarms/yr. Seeds of the
  population: short glitches (3111) median |t₀ error| 9.7 s (90% 35 s, 99% 65 s, one of 3150
  outside 2.6e3 s), Δv̂ within −8/+13% (16–84%), arm 99.8%; τ > 53 s (39): 2.9 s, Δv ±4%,
  τ ±5%. With the ρ ≈ 3e5 catalogue row kept (`--keep-all`): its three draws are subtracted
  first (t₀ to 2 ms), the 41 sidelobe events vanish and the year runs like the filtered one;
  every glitch with ρ > 12 is found.
- **A two-stage search** (the jaxmbhb design, measured for glitches): trigger excess power
  at 10 false alarms/yr, keep events whose seed has ρ_seed ≥ 5. Noise: 5.5% of 1564 noise
  events confirmed (160 yr) → 0.54 ± 0.06 false alarms/yr. Injections (`efficiency.py --mf
  --far 10`): ρ₅₀ = 6.17, 6.06, 6.15, 6.33, 6.65 at τ = 1…2e4 s — 9–15% below excess power
  alone at 1/yr, within 1.16–1.33 of the matched filter; confirmation loses 1 of 1470 found
  injections at ρ ≥ 8. The price: the confirmed list is no longer model-independent.

## Files

```
glitchsearch/
  grid.py      the year and the ladder of tilings (and why)
  simulate.py  frequency-domain noise, glitches on any of the six test masses, LPF catalogue years
  wdm.py       WDM kernel from the one-sided spectrum, whitening, robust renormalisation
  search.py    tile bank, chi^2 tails, clusters, cross-tiling events, Search
  far.py       false-alarm calibration on noise (FarCurve)
  design.py    predicted sensitivity from noiseless templates
  truth.py     what made each event, in simulations
  mfilter.py   matched filter: bank in tau x arm, Delta v analytic, t0 by FFT; the follow-up
  pipeline.py  search, follow up, subtract, search again
  noise.py     verbatim copy of jaxglitches/notebooks/noise.py (commit ee666d1)
  plotting.py  the paper's figure style
scripts/       design_tiling, calibrate_far, efficiency, run_search, followup (see Quick start)
tests/         the kernel against wdm_transform, chi^2 calibration, recovery, attribution,
               the matched filter, the follow-up across the year's join, the loop on a pair
```

## Limits, and what would come next

- **Periodic simulated year.** Like the jaxglitches examples, the noise is generated
  periodic, so the join of the year is harmless and the search wraps across it. Real data
  have a red-noise jump there (the paper's Sec. "A trap: short segments and red noise"), and
  the bins next to the edges would need masking or padding.
- **Very loud glitches.** The catalogue's inconsistent 2.1×10⁻⁸ m/s row, which the bootstrap
  draws a few times a year, reaches ρ ≈ 3×10⁵. Its wavelet sidelobes then clear threshold for
  days: with it kept (`run_search.py lpf --keep-all`), 41 pile-up events appear and 14
  extra loud glitches are swallowed. `lpf_catalogue` drops that row and the 6.5×10⁻¹⁰ one,
  as the paper does. The loop's loud-first stage handles it if such glitches are real.
- **Close pairs** (< ~30 min apart) come out as one event in a single pass; the loop finds
  the second glitch once the first is subtracted. Its seeds are single-glitch fits made in
  sequence, not a joint fit: a pair's first seed is fitted with the second still in the data.
  A joint re-fit of overlapping seeds, or reversible jump, in the next stage would fix that.
- **Only instrument noise and glitches.** Galactic binaries would enter as stationary lines,
  which `--robust` absorbs. Massive black hole binaries are transients too and would
  trigger, so a real pipeline needs a glitch-versus-GW classification step, and the
  band top and seed tiling are a start on it. Non-stationary noise fits naturally: a
  running median along the time axis instead of one median per channel.
- **Equal arms, TDI-1.** TDI-2 changes nothing in principle, since whitening absorbs the
  transfer function and its first null, at 1/(4L) ≈ 30 mHz, lies above the band. Unequal
  arms would break the test-mass invariance of A + E and bring T back.
- **Sensitivity.** The two-stage search above closes part of the factor 1.3–1.4 against
  matched filtering; tuning the trigger and confirmation thresholds jointly (at
  ρ_seed ≥ 5.5 the confirmed noise rate is 0.07/yr) could close more. Weighting each
  channel of a tile by the expected power profile of a τ class (a locally optimal quadratic
  statistic) would do it without templates.

# Toward a search block in the APC global fit

**Status (2026-10-14): not plugged in.** The merge request
[lisa-apc/global-fit!186](https://gitlab.in2p3.fr/lisa-apc/global-fit/-/merge_requests/186)
(A. Vajpeyi, branch `glitch-block` of `avi.vajpeyi/global-fit`, open) adds
`globalfit.glitch`. Its `GlitchBlock` fits one glitch, starting from a detection estimate it
is given (`source = {t0, Deltav, tau}`). That estimate is what this search produces. This
note tracks what already matches and what a search block would still need. The MR is not
merged, so its interface may still change.

## What already matches

- **jaxglitches.** Both use the PyPI release. The global fit pins `jaxglitches==0.1.0` (its
  `glitch` extra); this package asks for `>=0.1.0,<0.2`. The CI runs the tests with the
  global fit's lock (Python 3.13, jax 0.10.2, numpy 2.3.5, scipy 1.18.1), with the latest
  versions on Python 3.12, and weekly against jaxglitches `main`.
- **Dependencies.** Nothing the global fit does not already install: jax, numpy, scipy,
  jaxglitches. The WDM kernel is our own (`wdm_transform` is a test-only dependency), and
  matplotlib is needed only by `scripts/`.
- **Parameters.** [t0, Deltav, tau], in jaxglitches' order, names and units.
  `Seed.source(time_origin)` returns `{"t0", "Deltav", "tau", "link"}`. These are the keys
  of `GlitchBlock`'s `source` and of the columns of `glitch_cat.h5`, with t0 made absolute
  (the global fit's `time_origin` is the epoch of the window's first sample).
- **Fourier and noise normalisation.** Both use continuous Fourier amplitudes (`dt * rfft`)
  and the bin variance S T_obs / 2. The global fit writes it as S_psd / (2 df) after
  Cholesky whitening. SNRs carry over with no rescaling.
- **Links.** `glitchsearch.LINKS` follows jaxglitches' injection points and its lisaorbits
  link order, (12, 23, 31, 13, 32, 21). That is the order of the global fit's `ltt_<link>`
  catalogue columns (`tests/test_jaxglitches.py`).

## Gaps, in the order a block would meet them

1. **Data model.** The search assumes a periodic simulated year: equal arms, TDI-1, A + E
   with diagonal noise, dt = 60 s, N = 2^19. The global fit hands over XYZ, TDI-2 by
   default, arms frozen at the glitch epoch, an (F, 3, 3) covariance from `noise.cov`, and
   finite windows sampled at 0.25–5 s. A block would need:
   - whitening with the full covariance, either by Cholesky as `GlitchLikelihood` does or
     by eigen-decomposition as in jaxmbhb's `time_freq_ll`, keeping the two most sensitive
     whitened channels in place of A and E;
   - the spectrum cropped near 8.33 mHz, with the bin count chosen so that
     N = 2 (F − 1) is a multiple of 2 × the finest nt (2^18 for the default ladder).
     `Grid(dt, n)` takes any such n. For a 365.25-d window, N = 2^19 means a cut at
     dt = 60.19 s; the data are never resampled, only cropped in frequency;
   - a check of the A + E test-mass invariance. It is exact only for equal arms, so with
     unequal arms either measure the loss or bring T back.
2. **Edges.** The search wraps across the year's join. Real windows are not periodic, and
   the red-noise jump at the edges needs the bins next to them masked, or the record
   padded. `GBSearchBlock` tapers its input with a Planck(0.05) window. A taper breaks the
   iid N(0, 1) pixels near the edges, and `GlitchBlock` refuses one unless the template
   carries it too.
3. **False-alarm calibration.** `results/far_calibration*.npz` holds for N = 2^19, the
   default ladder and the SciRD noise. A block on another grid, or with fitted noise, needs
   its own calibration (`calibrate_far.py`), or the median-renormalised `--robust` mode. The
   threshold moves with the number of tiles.
4. **Seeds into `GlitchBlock`.** The block fits a glitch on link 12 with Deltav > 0
   (log-uniform prior). Seeds come on any of the six test masses, with either sign (LPF
   polarity is random, and `jaxglitches.run_catalog` is signed). The arm of a seed is
   reliable (99.8% in five LPF-like years); its test mass within the arm is not (50/50 in
   A + E). The block would need to take the link (or the arm) and the polarity. One route
   is the equal-arm relabelling in `simulate._LINK_MAP`; another is link-generic templates
   in jaxglitches. Also, a block fits one glitch: it reads row 0 of the previous catalogue.
   An LPF-like year gives ~600 events, so it would need one fit per seed or a multi-source
   block. Pairs closer than ~30 min want a joint fit.
5. **Follow-up templates.** `mfilter` refines seeds with equal-arm TDI-1 templates
   (`jg.clean_signal_f`). In the pipeline, the refinement should use the response the
   block fits with (`tdi=2`, and `clean_signal_f_unequal` with `link_ltt` at the seed's
   epoch). Otherwise the seed's t0 and Deltav carry a model error. Only the bank changes:
   the analytic Deltav and the FFT over t0 do not depend on the response.
6. **Other transients.** MBHBs make excess power too. The search should run after MBHB
   subtraction (the `sub` inputs of `GBSearchBlock` do this with `mbhb_cat.h5`), and before
   `GlitchBlock` there should be a glitch-versus-GW step: band top, seed tiling, the
   (A, E) direction against the three arms, the follow-up residual.
7. **Block shape.** `GBSearchBlock` (`globalfit/root/gbsearch_block.py`) is the template.
   Its inputs are the combined TDI minus the other sources' catalogues, and its output is a
   catalogue written by the source's `Generator.save_catalog`. For glitches that would be
   `GlitchGenerator.save_catalog(path, [seed.source(t_ini) | metadata, ...])`, where the
   metadata is `tdi_generation` and `arm_time` or `ltt_<link>`.

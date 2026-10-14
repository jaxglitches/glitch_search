"""Calibrate the false-alarm rate on noise-only years.

Runs the whole pipeline -- whitening, the five tilings, every tile, clustering -- on noise
alone and counts the events it produces at each threshold of a grid, rerunning the
clustering per threshold (see glitchsearch/far.py for why one loose run will not do). That
is the event rate the search produces in noise, which the Bonferroni bound n_trials * p
overestimates by the correlation between tiles (overlapping in time, nested in frequency,
repeated across tilings). Also checks that single-tile p-values are uniform, the premise
of both.

    python scripts/calibrate_far.py [--years 200] [--robust]
    python scripts/calibrate_far.py --mf [--years 400]

Writes results/far_calibration[_robust].npz (loaded by the other scripts when present) and
figures/far_calibration[_robust].pdf. About 1 min per 100 noise years on a CPU.

--mf calibrates the matched-filter bank of glitchsearch/mfilter.py instead, the same way:
events per threshold on noise years, with log p = log P(chi^2_1 > z^2) of the loudest
template, and writes results/far_calibration_mf.npz. About 2 min per 100 noise years.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import jax.random as jr  # noqa: E402

from glitchsearch import (Grid, Search, FarCurve, noise_event_counts, threshold_grid,  # noqa: E402
                          noise_fd)
from glitchsearch.plotting import plt, save, C, COL_IN  # noqa: E402

RES, FIG = HERE.parent / "results", HERE.parent / "figures"


def tile_uniformity(search, n=5, seed=500):
    """Fraction of single-tile p-values below 1e-2, 1e-3, 1e-4 over n noise years."""
    counts, total = np.zeros(3), 0
    for i in range(n):
        scans = search.scan(noise_fd(jr.PRNGKey(seed + i), search.grid))
        lp = np.concatenate([sc.logp.ravel() for sc in scans])
        counts += [(lp < np.log(q)).sum() for q in (1e-2, 1e-3, 1e-4)]
        total += lp.size
    return counts / total, total


def calibrate_mf(grid, n_years):
    """The matched filter's false-alarm rate against its per-template log p threshold."""
    from glitchsearch import mfilter
    mf = mfilter.MatchedFilter(grid, cache=RES / "mf_bank.npz")
    print(mf)
    bonf = lambda far: np.log(far * grid.years / mf.n_trials)            # noqa: E731
    thresholds = np.linspace(bonf(1e-2), bonf(500.0), 36)
    t1 = time.time()
    counts, years = mfilter.noise_event_counts(
        mf, n_years, thresholds,
        progress=lambda i, c: print(f"  {i + 1:4d} noise years", flush=True)
        if (i + 1) % 50 == 0 else None)
    print(f"{n_years} noise years in {time.time() - t1:.0f} s")
    curve = FarCurve(thresholds, counts, years, mf.n_trials, grid.years)
    RES.mkdir(exist_ok=True)
    curve.save(RES / "far_calibration_mf.npz")
    print(f"  wrote {RES / 'far_calibration_mf.npz'}")
    print(f"n_trials = {mf.n_trials:.4g} (template x onset samples)/yr, n_eff = {curve.n_eff:.4g}")
    print(f"{'FAR [1/yr]':>11} {'log10 p':>9} {'|z|':>7}")
    for far in (0.1, 1.0, 10.0):
        x = curve.threshold(far)
        print(f"{far:11.2g} {x / np.log(10):9.3f} {np.sqrt(mfilter._chi2_1_isf(x)):7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=200)
    ap.add_argument("--robust", action="store_true", help="median-renormalised channels")
    ap.add_argument("--mf", action="store_true", help="calibrate the matched filter instead")
    args = ap.parse_args()
    if args.mf:
        calibrate_mf(Grid(), args.years)
        return
    tag = "_robust" if args.robust else ""
    grid = Grid()
    search = Search(grid, robust=args.robust)
    print(grid, search, sep="\n")

    frac, total = tile_uniformity(search)
    print(f"\nsingle-tile p-values over {total:.3g} noise tiles: fraction below 1e-2, 1e-3, 1e-4 = "
          + ", ".join(f"{f:.3e}" for f in frac))

    t1 = time.time()
    thresholds = threshold_grid(search)
    counts, years = noise_event_counts(
        search, args.years, thresholds,
        progress=lambda i, c: print(f"  {i + 1:4d} noise years", flush=True)
        if (i + 1) % 100 == 0 else None)
    print(f"{args.years} noise years in {time.time() - t1:.0f} s")
    curve = FarCurve(thresholds, counts, years, search.n_trials, grid.years)
    RES.mkdir(exist_ok=True)
    curve.save(RES / f"far_calibration{tag}.npz")
    print(f"  wrote {RES / f'far_calibration{tag}.npz'}")

    print(f"\nn_trials = {search.n_trials:.4g} tiles/yr, n_eff = {curve.n_eff:.4g} "
          f"(correlation discount {curve.n_eff / search.n_trials:.3f})")
    print(f"{'FAR [1/yr]':>11} {'log10 p, Bonferroni':>20} {'log10 p, calibrated':>20}")
    for far in (0.01, 0.1, 1.0, 10.0):
        print(f"{far:11.2g} {search.log_p_bonferroni(far) / np.log(10):20.3f} "
              f"{curve.threshold(far) / np.log(10):20.3f}")

    fig, ax = plt.subplots(figsize=(COL_IN, 2.4))
    x = -curve.x / np.log(10)
    ok = curve.counts > 0
    ax.errorbar(x[ok], curve.counts[ok] / years, np.sqrt(curve.counts[ok]) / years, fmt="o",
                ms=3, color=C["blue"], lw=0.8, label=f"measured, {args.years} noise years")
    xs = np.linspace(x.min(), x.max() + 1.5, 50)
    ax.semilogy(xs, search.n_trials * 10 ** (-xs) / grid.years, color=C["grey"], ls="--",
                label="Bonferroni bound, $n_{\\rm trials}\\,p$")
    ax.semilogy(xs, curve.rate0 * 10 ** (-xs), color=C["red"], ls=":",
                label=f"tail law, $n_{{\\rm eff}} = {curve.n_eff / search.n_trials:.2f}\\,"
                      f"n_{{\\rm trials}}$")
    ax.set_yscale("log")
    ax.axhline(1.0, color=C["black"], lw=0.6)
    ax.set(xlabel=r"threshold $-\log_{10} p$", ylabel="noise events per year",
           title="false-alarm rate of the search" + (" (robust)" if args.robust else ""))
    ax.legend(loc="upper right")
    save(fig, FIG / f"far_calibration{tag}.pdf")


if __name__ == "__main__":
    main()

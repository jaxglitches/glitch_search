"""False-alarm rates measured on noise, rather than bounded.

Each tile's p-value is exact, but the search takes the most significant of ~1e6 tiles a
year, which overlap in time (k = 1 and 2 share bins), nest in frequency (every band is a
low-pass band), and repeat across five tilings. The Bonferroni rate n_trials * p per year
bounds the event rate from above. `FarCurve` replaces it with the rate of events the full
pipeline produces in noise-only years, *run at each threshold* of a grid.

That last point is not a detail. Clustering is not nested in the threshold: at a loose
threshold, fine-level noise tiles are common enough that a coarse cluster usually overlaps
one of them and is filed as its support, while at the operating threshold the same coarse
cluster is an event of its own. Counting the events of one loose run above a stricter
threshold therefore undercounts, by 25-35% at the rates of interest here. So the tile
scores of each noise year are computed once, and the clustering is rerun at every
threshold of the grid.

Beyond the most stringent threshold that still sees >= MIN_COUNT noise events, the curve
follows the tail law of the bound, FAR = rate0 * p, with rate0 fitted there;
n_eff = rate0 x (record length) is the effective number of independent tiles per record.
"""
from pathlib import Path

import numpy as np
import jax.random as jr

from . import simulate

MIN_COUNT = 20


class FarCurve:
    """Measured false-alarm rate (per Julian year) of the search against its log p threshold."""

    def __init__(self, thresholds, counts, years: float, n_trials: int, record_years: float):
        order = np.argsort(thresholds)
        self.x = np.asarray(thresholds, dtype=float)[order]      # most stringent first
        self.counts = np.asarray(counts, dtype=float)[order]
        self.years = float(years)
        self.n_trials = int(n_trials)
        self.record_years = float(record_years)
        self.rate = np.maximum.accumulate(self.counts) / self.years   # monotone in x
        ok = np.nonzero(self.counts >= MIN_COUNT)[0]
        if ok.size == 0:
            raise ValueError("no threshold of the grid saw enough noise events; loosen it")
        fit = ok[:3]                                             # the three most stringent
        self.rate0 = float(np.exp(np.mean(np.log(self.rate[fit]) - self.x[fit])))
        self.n_eff = self.rate0 * self.record_years
        self.x_tail = float(self.x[ok[0]])

    def far(self, logp):
        """Noise events per year produced by running the search at threshold `logp`."""
        logp = np.asarray(logp, dtype=float)
        inside = np.exp(np.interp(logp, self.x, np.log(np.maximum(self.rate, 1e-300))))
        return np.where(logp < self.x_tail, self.rate0 * np.exp(logp), inside)

    def log10_far(self, logp):
        """log10 of `far`, finite however significant the event."""
        logp = np.asarray(logp, dtype=float)
        inside = np.interp(logp, self.x, np.log10(np.maximum(self.rate, 1e-300)))
        return np.where(logp < self.x_tail, np.log10(self.rate0) + logp / np.log(10.0), inside)

    def threshold(self, far_per_year: float) -> float:
        """The log p threshold at which the search produces far_per_year noise events."""
        x_tail_rate = self.rate0 * np.exp(self.x_tail)
        if far_per_year <= x_tail_rate:
            return float(np.log(far_per_year / self.rate0))
        if far_per_year > self.rate[-1]:
            raise ValueError(f"FAR {far_per_year}/yr is above the loosest calibrated rate "
                             f"({self.rate[-1]:.3g}/yr); recalibrate with a looser grid")
        i = np.nonzero(self.x >= self.x_tail)[0]
        return float(np.interp(np.log(far_per_year), np.log(self.rate[i]), self.x[i]))

    def save(self, path):
        np.savez(path, x=self.x, counts=self.counts, years=self.years,
                 n_trials=self.n_trials, record_years=self.record_years)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["x"], z["counts"], float(z["years"]), int(z["n_trials"]),
                   float(z["record_years"]))


def threshold_grid(search, far_lo=1e-2, far_hi=300.0, n=30):
    """log p thresholds whose Bonferroni rates span [far_lo, far_hi] per year."""
    return np.linspace(search.log_p_bonferroni(far_lo), search.log_p_bonferroni(far_hi), n)


def noise_event_counts(search, n_realisations: int, thresholds, seed: int = 10_000,
                       progress=None):
    """Number of events the search produces on noise-only years at each threshold, and the
    total span in years. Tiles are scored once per year; clustering is rerun per threshold."""
    counts = np.zeros(len(thresholds))
    for i in range(n_realisations):
        scans = search.scan(simulate.noise_fd(jr.PRNGKey(seed + i), search.grid))
        counts += [len(search.events(scans, x)) for x in thresholds]
        if progress is not None:
            progress(i, counts)
    return counts, n_realisations * search.grid.years


def load_calibration(results_dir, robust: bool = False):
    """The FarCurve written by scripts/calibrate_far.py, or None if it has not been run."""
    path = Path(results_dir) / f"far_calibration{'_robust' if robust else ''}.npz"
    return FarCurve.load(path) if path.exists() else None

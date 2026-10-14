"""Excess-power search for glitches along the time axis of the WDM plane.

The statistic
-------------
The data are whitened in the frequency domain and transformed to every tiling of the
ladder (`wdm.py`). The pixels of A and E are then independent N(0, 1) under Gaussian
stationary noise, and a glitch deposits its rho^2 among them. T is left out: a test-mass
glitch puts < 0.7% of its rho^2 there, so T would add degrees of freedom and no signal.

A glitch is short and broadband, so in each tiling it is a vertical stripe: excess power
in one or two time bins, from the lowest channels up to a frequency set by its knee. The
search slides a small bank of tiles along the time axis -- k in {1, 2} consecutive time bins
by the low-pass band of channels 1..m_hi, m_hi on a sqrt(2) ladder up to 6 mHz -- and sums
the pixel power in each,

    E(n; k, m_hi) = sum_{n'=n}^{n+k-1} sum_{m=1}^{m_hi} (w_A^2 + w_E^2)_{n'm},

which under noise is chi^2 with 2 k m_hi degrees of freedom, and under a glitch non-central
chi^2 with non-centrality equal to the rho^2 the tile captures. Every tile therefore has an
exact p-value, and nothing about the glitch beyond "short, starting at low frequency" is
assumed: no template, no phase, no grid in t0. The 50%-detection SNR at a false-alarm rate
of 1/yr is predicted at 7.1-8.4 from tau = 0.1 s to 2.3e4 s and measured on injections at
6.7-7.2 (the prediction credits only the best tile); an ideal matched filter over the same
year would need 4.5-5.4 (Rice estimate), so assuming only compactness costs a factor
1.3-1.8 in SNR (`scripts/design_tiling.py`, `scripts/efficiency.py`).

The bands are low-pass, [1, m_hi], because lower edges m_lo > 1 detected nothing better at
any duration -- the extra trials cost more than the channels they drop -- so they were
left out rather than kept for generality (`design_tiling.py --banks`).

From tiles to events
--------------------
Tiles above threshold are merged, level by level, into clusters of contiguous time bins.
Clusters are then associated across levels from the finest down. A coarser cluster that
overlaps no event starts a new one (a glitch too long for the finer tilings); one that
overlaps events is attached to them as support, unless its -log p is at least
`merge_factor` times that of every event it overlaps, in which case it absorbs them. That
last rule is what lets a long glitch, fragmented into weak clusters at fine resolution,
come back as one event, while two short glitches that share a 17-hour bin -- routine at the
LPF rate of ~1.3 detectable glitches a day -- stay two.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import special

from . import wdm
from .grid import LADDER

LN10 = np.log(10.0)


# ---------------------------------------------------------------------------
# chi^2 tail, finite far beyond where scipy underflows
# ---------------------------------------------------------------------------

def _log_q_cf(a, z, n_iter=400):
    """log Q(a, z), Q the regularised upper incomplete gamma, by the Lentz continued
    fraction (Numerical Recipes, gcf). Valid for z > a + 1, where it converges fast."""
    tiny = 1e-300
    b = z + 1.0 - a
    c = np.full_like(z, 1.0 / tiny)
    d = 1.0 / b
    h = d.copy()
    for i in range(1, n_iter + 1):
        an = -i * (i - a)
        b = b + 2.0
        d = an * d + b
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = b + an / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        h = h * d * c
    return -z + a * np.log(z) - special.gammaln(a) + np.log(h)


def chi2_logsf(x, dof):
    """Natural log of P(chi^2_dof > x). Below p ~ 1e-300, where scipy returns -inf, the
    continued fraction takes over, so a glitch of rho ~ 1e3 still gets a finite number."""
    x = np.asarray(x, dtype=float)
    dof = np.broadcast_to(np.asarray(dof, dtype=float), x.shape)
    with np.errstate(divide="ignore"):
        out = np.log(special.gammaincc(0.5 * dof, 0.5 * x))
    far = ~np.isfinite(out) | (out < -690.0)
    if np.any(far):
        out = np.array(out)
        out[far] = _log_q_cf(0.5 * dof[far], 0.5 * x[far])
    return out


# ---------------------------------------------------------------------------
# The tile bank
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """A tile: k consecutive time bins by channels 1 .. m_hi."""
    k: int
    m_hi: int


def bank_for(level, f_top: float = 6e-3, ratio: float = np.sqrt(2.0), ks=(1, 2)):
    """Low-pass bands [1, m_hi] with m_hi on a geometric ladder of the given ratio, up to
    the channel nearest f_top, each at k time bins."""
    m_top = max(1, min(level.n_chan, int(np.floor(f_top / level.delta_f))))
    n = max(2, int(np.log(m_top) / np.log(ratio)) + 2)
    his = np.unique(np.round(np.geomspace(1, m_top, n)).astype(int))
    return [Shape(int(k), int(h)) for k in ks for h in his]


# ---------------------------------------------------------------------------
# Scanning one tiling
# ---------------------------------------------------------------------------

@dataclass
class LevelScan:
    """Every tile of one tiling: energy[s, n] and logp[s, n] for shape s starting at bin n."""
    level: object
    shapes: list
    dof: np.ndarray          # (S,)
    power: np.ndarray        # (nt, n_chan) pixel power, summed over the TDI channels
    energy: np.ndarray       # (S, nt)
    logp: np.ndarray         # (S, nt)


def scan_level(power, level, shapes, n_tdi: int = 2) -> LevelScan:
    cum = np.cumsum(power, axis=1)
    energy = np.empty((len(shapes), level.nt))
    for s, sh in enumerate(shapes):
        col = cum[:, sh.m_hi - 1]
        energy[s] = sum(np.roll(col, -j) for j in range(sh.k))   # bins n .. n+k-1, periodic
    dof = np.array([n_tdi * sh.k * sh.m_hi for sh in shapes], dtype=float)
    return LevelScan(level, shapes, dof, power, energy, chi2_logsf(energy, dof[:, None]))


# ---------------------------------------------------------------------------
# Clusters and events
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    """Contiguous significant time bins of one tiling, represented by their best tile."""
    level: object
    t_lo: float              # interval [t_lo, t_hi) in s; may run past 0 or T_obs (periodic)
    t_hi: float
    shape: Shape
    n0: int
    logp: float
    energy: float
    dof: float
    t_hat: float             # excess-power centroid in time, in [0, T_obs)


@dataclass
class Event:
    """A glitch candidate. `core` holds the clusters that define it (its seed and any
    fragments it absorbed); `support` the coarser clusters it shares with other events."""
    core: list
    support: list = field(default_factory=list)
    log10_far: float = np.nan        # false alarms per year of the search run at this log p

    @property
    def far(self) -> float:
        return float(10.0 ** self.log10_far)

    @property
    def seed(self) -> Cluster:
        return min(self.core, key=lambda c: c.logp)

    @property
    def logp(self) -> float:
        return self.seed.logp

    @property
    def log10p(self) -> float:
        return self.seed.logp / LN10

    @property
    def t_hat(self) -> float:
        return self.seed.t_hat

    @property
    def rho_ep(self) -> float:
        """Excess-power SNR of the seed tile, sqrt(E - dof): an estimate of the part of rho
        the tile captures: 0.7-0.8 of the optimal SNR for glitches found at rho >= 10
        (scripts/efficiency.py)."""
        s = self.seed
        return float(np.sqrt(max(s.energy - s.dof, 0.0)))

    @property
    def f_hi(self) -> float:
        """Upper edge of the seed tile's band: a proxy for the knee, hence for tau."""
        s = self.seed
        return (s.shape.m_hi + 0.5) * s.level.delta_f

    @property
    def levels(self):
        return sorted({c.level.nt for c in self.core + self.support}, reverse=True)

    def overlaps(self, cl: Cluster, t_obs: float) -> bool:
        return any(_circ_overlap(c.t_lo, c.t_hi, cl.t_lo, cl.t_hi, t_obs) for c in self.core)

    def summary(self) -> dict:
        s = self.seed
        return dict(t_hat=self.t_hat, log10p=self.log10p, log10_far=self.log10_far,
                    far=self.far, rho_ep=self.rho_ep,
                    nt=s.level.nt, delta_t=s.level.delta_t, k=s.shape.k, m_hi=s.shape.m_hi,
                    f_hi=self.f_hi, t_lo=s.t_lo, t_hi=s.t_hi, levels=self.levels)


def _circ_overlap(a_lo, a_hi, b_lo, b_hi, period):
    return any(a_lo < b_hi + s and b_lo + s < a_hi for s in (-period, 0.0, period))


def _circular_runs(mask, gap: int):
    """Runs of True on a circle, joined across gaps of at most `gap` False bins.
    Returns ([(start, length)], label) with label[n] the run of bin n, or -1."""
    nt = mask.size
    label = np.full(nt, -1)
    if not mask.any():
        return [], label
    if mask.all():
        label[:] = 0
        return [(0, nt)], label
    shift = int(np.argmin(mask))                       # an empty bin: cut the circle there
    d = np.diff(np.concatenate([[0], np.roll(mask, -shift).astype(np.int8), [0]]))
    runs = []
    for s, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
        if runs and s - runs[-1][1] <= gap:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    if len(runs) > 1 and runs[0][0] + nt - runs[-1][1] <= gap:   # join across the cut
        runs[0][0] = runs[-1][0] - nt
        runs.pop()
    out = []
    for r, (s, e) in enumerate(runs):
        label[(np.arange(s, e) + shift) % nt] = r
        out.append(((s + shift) % nt, min(e - s, nt)))
    return out, label


def _centroid(scan: LevelScan, s: int, n0: int, n_tdi: int) -> float:
    """Time centroid of the excess power in the band of shape s, over the tile's bins and
    one bin either side (periodic), in [0, T_obs)."""
    lev, sh = scan.level, scan.shapes[s]
    rel = np.arange(-1, sh.k + 1)
    e = scan.power[(n0 + rel) % lev.nt, :sh.m_hi].sum(axis=1) - n_tdi * sh.m_hi
    wgt = np.clip(e, 0.0, None)
    c = n0 + (rel @ wgt / wgt.sum() if wgt.sum() > 0 else 0.5 * (sh.k - 1))
    return float((c * lev.delta_t) % (lev.nt * lev.delta_t))


def level_clusters(scan: LevelScan, log_p_thr: float, gap_bins: int = 1, n_tdi: int = 2):
    """Significant tiles of one tiling, merged into clusters of contiguous time bins."""
    lev = scan.level
    s_idx, n_idx = np.nonzero(scan.logp < log_p_thr)
    if s_idx.size == 0:
        return []
    ks = np.array([sh.k for sh in scan.shapes])[s_idx]
    covered = np.zeros(lev.nt, bool)
    for j in range(ks.max()):
        covered[(n_idx[ks > j] + j) % lev.nt] = True
    runs, label = _circular_runs(covered, gap_bins)
    tile_run = label[n_idx]
    out = []
    for r, (b0, length) in enumerate(runs):
        sel = np.nonzero(tile_run == r)[0]
        i = sel[np.argmin(scan.logp[s_idx[sel], n_idx[sel]])]
        s, n0 = int(s_idx[i]), int(n_idx[i])
        out.append(Cluster(level=lev, t_lo=(b0 - 0.5) * lev.delta_t,
                           t_hi=(b0 + length - 0.5) * lev.delta_t, shape=scan.shapes[s],
                           n0=n0, logp=float(scan.logp[s, n0]),
                           energy=float(scan.energy[s, n0]), dof=float(scan.dof[s]),
                           t_hat=_centroid(scan, s, n0, n_tdi)))
    return out


def build_events(clusters_by_level, t_obs: float, merge_factor: float = 2.0):
    """Associate clusters across tilings, finest first (see the module docstring)."""
    flat = sorted((c for cl in clusters_by_level for c in cl),
                  key=lambda c: (-c.level.nt, c.logp))
    events = []
    for cl in flat:
        hits = [e for e in events if e.overlaps(cl, t_obs)]
        if not hits:
            events.append(Event(core=[cl]))
        elif -cl.logp >= merge_factor * max(-e.logp for e in hits):
            merged = Event(core=[cl] + [c for e in hits for c in e.core],
                           support=[c for e in hits for c in e.support])
            events = [e for e in events if not any(e is h for h in hits)] + [merged]
        else:
            for e in hits:
                e.support.append(cl)
    return sorted(events, key=lambda e: e.t_hat)


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

class Search:
    """Multi-resolution WDM excess-power search over one year of A, E data.

    grid         : glitchsearch.grid.Grid.
    ladder       : WDM time-bin counts to scan (default grid.LADDER; grid.SINGLE for one).
    f_top, ks    : tile bank: bands [1, m_hi] up to f_top, k consecutive time bins.
    robust       : renormalise each WDM channel by its median power (unknown or
                   mis-estimated PSD, stationary lines); off when the PSD is known.
    calibration  : optional far.FarCurve measured on noise; without it false-alarm rates
                   are Bonferroni bounds, n_trials * p per year.
    """

    def __init__(self, grid, ladder=LADDER, f_top: float = 6e-3, ks=(1, 2),
                 channels=(0, 1), robust: bool = False, calibration=None):
        self.grid = grid
        self.levels = grid.levels(ladder)
        self.banks = [bank_for(lev, f_top=f_top, ks=ks) for lev in self.levels]
        self.channels = tuple(channels)
        self.n_tdi = len(self.channels)
        self.robust = robust
        self.calibration = calibration

    @property
    def n_trials(self) -> int:
        """Number of tiles scanned in a year: sum over tilings of nt x bank size."""
        return int(sum(lev.nt * len(b) for lev, b in zip(self.levels, self.banks)))

    def log_p_bonferroni(self, far_per_year: float) -> float:
        """Per-tile log p at which the tiles alone would give far_per_year false alarms.
        Conservative, since tiles overlap in time and nest in frequency."""
        return float(np.log(far_per_year * self.grid.years / self.n_trials))

    def threshold(self, far_per_year: float) -> float:
        if self.calibration is not None:
            return self.calibration.threshold(far_per_year)
        return self.log_p_bonferroni(far_per_year)

    def log10_far_of(self, logp: float) -> float:
        """log10 of the false-alarm rate (per year) of the search run at threshold logp:
        measured if a FarCurve is attached, the Bonferroni bound otherwise."""
        if self.calibration is not None:
            return float(self.calibration.log10_far(logp))
        return float((np.log(self.n_trials / self.grid.years) + logp) / LN10)

    def scan(self, data_fd, robust=None):
        """Whiten, transform to every tiling, and evaluate every tile. data_fd: (F, 3).
        `robust` overrides the search's setting; a noiseless signal must be scanned with
        robust=False, since its median power is ~0 and says nothing about the noise."""
        robust = self.robust if robust is None else robust
        Y = wdm.whiten(data_fd, self.grid, self.channels)
        return [scan_level(wdm.pixel_power(Y, lev, robust), lev, bank, self.n_tdi)
                for lev, bank in zip(self.levels, self.banks)]

    def events(self, scans, log_p_thr: float, gap_bins: int = 1, merge_factor: float = 2.0):
        clusters = [level_clusters(sc, log_p_thr, gap_bins, self.n_tdi) for sc in scans]
        events = build_events(clusters, self.grid.t_obs, merge_factor)
        for e in events:
            e.log10_far = self.log10_far_of(e.logp)
        return events

    def run(self, data_fd, far_per_year: float = 1.0, log_p_thr=None, **kw):
        """Scan and return (events, scans). The threshold is log_p_thr if given, else the
        per-tile log p matching far_per_year (calibrated if a FarCurve is attached)."""
        scans = self.scan(data_fd)
        thr = self.threshold(far_per_year) if log_p_thr is None else log_p_thr
        return self.events(scans, thr, **kw), scans

    def significance(self, scans):
        """Per tiling, the largest -log10 p among the tiles starting at each time bin."""
        return [(-sc.logp.min(axis=0) / LN10) for sc in scans]

    def __repr__(self):
        rows = ", ".join(f"{lev.nt}:{len(b)}" for lev, b in zip(self.levels, self.banks))
        return f"Search(nt:shapes = {rows}; {self.n_trials:.3g} tiles/yr)"

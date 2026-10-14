"""Matched filtering for glitches: the yardstick for the excess-power search, and the
follow-up that turns its events into parameter seeds.

The idea is the one the MBHB search of jaxmbhb is built on (S. Babak's amplitude- and
time-maximised likelihood on the `mbhbh_search` branch; J. Perret's two-stage
`MBHBSeeker`, now in lisa-apc/geemoo-mbhb-detection): maximise the likelihood analytically
over every parameter it depends on linearly or through a time shift, and search only the
rest. A glitch has three parameters and a test mass, and two of the three go analytically:

    h_c(f; t0, Dv, tau, l) = Dv * u_c(f; tau, l) * exp(-2 pi i f t0).

With whitened data Y_ck = d_ck / sigma_ck and template U_ck = u_ck / sigma_ck,

    N(t0) = (d|u_t0) = 2 Re sum_ck conj(U_ck) Y_ck exp(2 pi i f_k t0),   M = (u|u),

so Dv_hat = N / M and the log-likelihood ratio maximised over Dv of either sign is
N^2 / (2M). N(t0) is one inverse FFT of the matched-filter spectrum C_k = sum_c Y_ck
conj(U_ck) for every t0 of the year at once; z = N / sqrt(M) is N(0, 1) under noise at a
fixed template, and equals rho at the true parameters.

What is left is a bank in tau and test mass, and both are smaller than they look.
  * Test mass: in A + E the two test masses of one arm give the same template to
    < 1e-4 mismatch (12 ~ 21, 23 ~ 32, 31 ~ 13), and the three arms sit 60 degrees apart
    in the (A, E) plane, so the bank needs one link per arm, ARMS.
  * Decay time: every glitch shorter than ~35 s is the same template to > 0.99, since its
    knee lies above the band; above ~100 s the family is self-similar in log tau.
    `bank_taus` places templates so that every tau in [0.01, 1e5] s matches its nearest
    template to MIN_MATCH = 0.98: 17 values, 51 templates with the arms.
  * Onset: t0 is resolved on a grid OVER = 2 times finer than dt. At dt = 60 s itself the
    correlation of a short glitch falls to 0.87 half a sample off its peak; at dt / 2 the
    worst case is 0.965 and the mean loss ~1%.

Two uses. `MatchedFilter.scan` / `.events` run the bank over the year as a search in its
own right, with its false-alarm rate calibrated on noise exactly like the excess-power
search (far.FarCurve on log p = log P(chi^2_1 > z^2)): the measured yardstick for what
knowing the waveform is worth. `followup` takes excess-power events and, within each
event's window, maximises the same statistic over the bank and then continuously over
(t0, log tau) with jax gradients, for both test masses of the best arm: a seed
(t0, Dv, tau, link) for jaxglitches, with the maximised SNR, as MBHBSeeker's Stage B does
for a binary.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from scipy import optimize

from . import simulate, wdm
from .search import chi2_logsf, _circular_runs
from .simulate import LINKS

jax.config.update("jax_enable_x64", True)

ARMS = ("12", "23", "31")          # one test mass per arm; the other is degenerate in A + E
PARTNER = {"12": "21", "23": "32", "31": "13", "21": "12", "32": "23", "13": "31"}
MIN_MATCH = 0.98
OVER = 2
TAU_RANGE = (0.01, 1e5)


# ---------------------------------------------------------------------------
# Templates and correlations
# ---------------------------------------------------------------------------

def unit_template(grid, tau, link="12", channels=(0, 1)):
    """Whitened template of a unit-Delta v glitch with onset t0 = 0, (len(channels), F)."""
    h = simulate.glitch_fd([0.0, 1.0, float(tau)], grid, link)
    return np.asarray(wdm.whiten(h, grid, channels))


def norm2(U):
    """(u|u) = 2 sum |U|^2 over channels and bins (DC is already zero)."""
    return 2.0 * float(np.sum(np.abs(U) ** 2))


def correlate(Y, U, over=OVER):
    """N(t) = (d|u_t) on the grid t_j = j dt / over, j = 0 .. over N - 1, for whitened data
    Y and template U, both (C, F). Zero-padding the spectrum interpolates the correlation
    exactly (it is band-limited to f_Nyq); with over >= 2 the grid's Nyquist bin is an
    interior bin of the padded transform and keeps its weight 2, as in jaxglitches."""
    if over < 2:
        raise ValueError("over must be >= 2 (see docstring)")
    C = np.sum(Y * np.conj(U), axis=0)
    n = over * 2 * (C.size - 1)
    Cz = np.zeros(n // 2 + 1, complex)
    Cz[:C.size] = C
    return n * np.fft.irfft(Cz, n=n)


def match(U1, U2, over=4):
    """Match of two whitened templates, maximised over a relative time shift and the sign."""
    r = correlate(U2, U1, over)
    return float(np.max(np.abs(r)) / np.sqrt(norm2(U1) * norm2(U2)))


def bank_taus(grid, min_match=MIN_MATCH, tau_range=TAU_RANGE, cache=None):
    """Decay times such that every tau in tau_range matches its nearest bank template to
    >= min_match (same arm, maximised over t0 and sign). Built greedily in log tau: from a
    template, bisect for the farthest tau it still covers, then for the next template that
    covers that tau from above. A few tens of seconds; cached to `cache` (an .npz) if given."""
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        if float(z["min_match"]) == min_match and tuple(z["tau_range"]) == tuple(tau_range):
            return z["taus"]
    lo, hi = np.log(tau_range[0]), np.log(tau_range[1])
    ucache = {}

    def U(x):
        if x not in ucache:
            ucache[x] = unit_template(grid, np.exp(x))
        return ucache[x]

    def farthest(x0, ok, x_max):
        """Largest x in (x0, x_max] with ok(x), by bisection (ok is monotone)."""
        if ok(x_max):
            return x_max
        a, b = x0, x_max
        for _ in range(30):
            m = 0.5 * (a + b)
            a, b = (m, b) if ok(m) else (a, m)
            if b - a < 1e-3:
                break
        return a

    taus = [lo]
    while True:
        edge = farthest(taus[-1], lambda x: match(U(taus[-1]), U(x)) >= min_match, hi)
        if edge >= hi:
            break
        nxt = farthest(edge, lambda x: match(U(x), U(edge)) >= min_match, hi)
        taus.append(nxt)
        if nxt >= hi:
            break
    taus = np.exp(np.array(taus))
    if cache is not None:
        np.savez(cache, taus=taus, min_match=min_match, tau_range=np.array(tau_range))
    return taus


def logp_of(z):
    """log P(chi^2_1 > z^2): the per-template p-value of a matched-filter output."""
    z = np.asarray(z, dtype=float)
    return chi2_logsf(z ** 2, np.ones_like(z))


# ---------------------------------------------------------------------------
# The matched filter as a search
# ---------------------------------------------------------------------------

@dataclass
class MFScan:
    """Best template at every onset time t_j = j dt / over: |z|, its sign and template index."""
    z: np.ndarray            # (over N,) signed z of the best template
    itpl: np.ndarray         # (over N,) index into MatchedFilter.templates
    dt: float                # grid spacing, s

    @property
    def absz(self):
        return np.abs(self.z)


@dataclass
class MFEvent:
    t_lo: float
    t_hi: float
    t0: float                # onset of the best template, s
    z: float                 # signed matched-filter SNR
    tau: float
    link: str

    @property
    def logp(self):
        return float(logp_of(self.z))


class MatchedFilter:
    """A bank of unit-Delta v glitch templates, maximised analytically over Delta v and by
    FFT over the onset. grid: glitchsearch.grid.Grid."""

    def __init__(self, grid, taus=None, arms=ARMS, channels=(0, 1), over=OVER,
                 min_match=MIN_MATCH, cache=None):
        self.grid = grid
        self.channels = tuple(channels)
        self.over = int(over)
        self.taus = np.asarray(bank_taus(grid, min_match, cache=cache) if taus is None else taus)
        self.arms = tuple(arms)
        self.templates = [(float(t), a) for t in self.taus for a in self.arms]
        self._U = None

    @property
    def U(self):
        """Whitened unit templates, normalised to (u|u) = 1: list of (C, F) arrays."""
        if self._U is None:
            self._U = []
            for tau, link in self.templates:
                u = unit_template(self.grid, tau, link, self.channels)
                self._U.append(u / np.sqrt(norm2(u)))
        return self._U

    @property
    def n_trials(self):
        return len(self.templates) * self.over * self.grid.n

    def scan(self, data_fd):
        Y = np.asarray(wdm.whiten(data_fd, self.grid, self.channels))
        best = np.zeros(self.over * self.grid.n)
        itpl = np.zeros(best.size, np.int32)
        for i, u in enumerate(self.U):
            z = correlate(Y, u, self.over)
            better = np.abs(z) > np.abs(best)
            best = np.where(better, z, best)
            itpl[better] = i
        return MFScan(best, itpl, self.grid.dt / self.over)

    def events(self, scan, log_p_thr, gap_s=3600.0):
        """Runs of onset times whose best |z| clears the threshold, joined across gaps of
        up to gap_s (periodic year); each is one event, at its loudest sample."""
        z_thr = np.sqrt(_chi2_1_isf(log_p_thr))
        above = scan.absz > z_thr
        runs, label = _circular_runs(above, int(round(gap_s / scan.dt)))
        out = []
        n = above.size
        for r, (b0, length) in enumerate(runs):
            idx = (b0 + np.arange(length)) % n
            idx = idx[above[idx]]
            j = int(idx[np.argmax(scan.absz[idx])])
            tau, link = self.templates[int(scan.itpl[j])]
            out.append(MFEvent(t_lo=b0 * scan.dt, t_hi=(b0 + length) * scan.dt, t0=j * scan.dt,
                               z=float(scan.z[j]), tau=tau, link=link))
        return out

    def run(self, data_fd, log_p_thr, **kw):
        sc = self.scan(data_fd)
        return self.events(sc, log_p_thr, **kw), sc

    def __repr__(self):
        return (f"MatchedFilter({len(self.taus)} decay times x {len(self.arms)} arms = "
                f"{len(self.templates)} templates, t0 grid dt/{self.over})")


def _chi2_1_isf(log_p):
    """z^2 at which P(chi^2_1 > z^2) = exp(log_p), for log_p down to ~-700."""
    from scipy import special
    return 2.0 * special.erfcinv(np.exp(log_p)) ** 2


def noise_event_counts(mf, n_realisations, thresholds, seed=10_000, progress=None, gap_s=3600.0):
    """As far.noise_event_counts, for the matched filter: events per threshold on noise years."""
    import jax.random as jr
    counts = np.zeros(len(thresholds))
    for i in range(n_realisations):
        sc = mf.scan(simulate.noise_fd(jr.PRNGKey(seed + i), mf.grid))
        counts += [len(mf.events(sc, x, gap_s)) for x in thresholds]
        if progress is not None:
            progress(i, counts)
    return counts, n_realisations * mf.grid.years


# ---------------------------------------------------------------------------
# Follow-up: from an event to a seed
# ---------------------------------------------------------------------------

@dataclass
class Seed:
    """Maximum-likelihood single-glitch parameters in an event's window."""
    t0: float
    dv: float
    tau: float
    link: str
    rho: float               # maximised matched-filter SNR, |N| / sqrt(M)
    z_bank: float            # the same on the bank, before the continuous refinement
    tau_bank: float
    sigma_t0: float          # Laplace widths of the profile likelihood (nan if not concave)
    sigma_logtau: float

    @property
    def params(self):
        return np.array([self.t0, self.dv, self.tau])

    @property
    def link_idx(self):
        return LINKS.index(self.link)

    def source(self, time_origin=0.0):
        """The seed under the names of jaxglitches' [t0, Deltav, tau] and of the APC global
        fit's glitch catalogue, with t0 made absolute: `time_origin` is the epoch of the
        record's first sample. Deltav keeps its sign and `link` the test mass; the global
        fit's glitch block fits link 12 with Deltav > 0 only (GLOBALFIT.md)."""
        return {"t0": self.t0 + float(time_origin), "Deltav": self.dv, "tau": self.tau,
                "link": self.link}


def _profile_nm(x, cross, norm, omega, t_scale):
    """N and M at onset x[0] t_scale (relative to the heterodyne reference already applied to
    the data) and tau = exp(x[1]).

    The decay time enters the template only through the factor q = 1 / (2 pi f tau - i)^2 of
    the glitch spectrum, so with V = U / q precomputed once (it does not depend on tau),
        N = 2 Re sum_k conj(q_k) cross_k exp(2 pi i f_k dt0),   cross_k = sum_c conj(V_ck) Y_ck,
        M = 2 sum_k |q_k|^2 norm_k,                             norm_k = sum_c |V_ck|^2,
    and an evaluation costs one complex exponential per bin instead of a template."""
    q = 1.0 / (omega * jnp.exp(x[1]) - 1j) ** 2
    phase = jnp.exp(1j * omega * (x[0] * t_scale))
    N = 2.0 * jnp.real(jnp.sum(jnp.conj(q) * cross * phase))
    M = 2.0 * jnp.sum(jnp.abs(q) ** 2 * norm)
    return N, M


def _profile(x, *args):
    """N^2 / M: twice the log-likelihood ratio with Delta v maximised out."""
    N, M = _profile_nm(x, *args)
    return N ** 2 / M


_profile_nm_jit = jax.jit(_profile_nm)
_profile_vg = jax.jit(jax.value_and_grad(_profile))
_profile_hess = jax.jit(jax.hessian(_profile))
_TAU_FREE = {}


def _tau_free(grid, link, channels):
    """V = U / q, the whitened unit template with its decay-time factor divided out, and
    sum_c |V_c|^2; cached per (grid, link, channels)."""
    key = (id(grid), link, tuple(channels))
    if key not in _TAU_FREE:
        omega = 2.0 * np.pi * grid.freq
        V = unit_template(grid, 1.0, link, channels) * (omega * 1.0 - 1j) ** 2
        _TAU_FREE[key] = (V, np.sum(np.abs(V) ** 2, axis=0))
    return _TAU_FREE[key]


def refine(Y, grid, t0, tau, link, channels=(0, 1)):
    """Continuous maximisation of N^2 / M over (t0, log tau) from a bank point (t0, tau), at
    a fixed test mass. The data are heterodyned to t0 first, so the optimiser works in a
    small offset. Returns (t0, tau, rho, dv, sigma_t0, sigma_logtau)."""
    V, norm = _tau_free(grid, link, channels)
    omega = 2.0 * np.pi * grid.freq
    cross = np.sum(np.conj(V) * np.asarray(Y), axis=0) * np.exp(1j * omega * float(t0))
    t_scale = max(60.0, min(float(tau), 3600.0))
    args = (jnp.asarray(cross), jnp.asarray(norm), jnp.asarray(omega), t_scale)

    def fun(x):
        v, g = _profile_vg(jnp.asarray(x), *args)
        return -float(v), -np.asarray(g)

    lo, hi = np.log(TAU_RANGE[0]), np.log(TAU_RANGE[1])
    w = 6.0 * 3600.0 / t_scale
    res = optimize.minimize(fun, np.array([0.0, np.log(tau)]), jac=True, method="L-BFGS-B",
                            bounds=[(-w, w), (lo, hi)],
                            options=dict(maxiter=300, ftol=1e-14, gtol=1e-9))
    x = jnp.asarray(res.x)
    N, M = (float(v) for v in _profile_nm_jit(x, *args))
    H = -0.5 * np.asarray(_profile_hess(x, *args))          # -d2 lnL in (x0, log tau)
    sig = np.full(2, np.nan)
    try:
        cov = np.linalg.inv(H)
        if np.all(np.diag(cov) > 0):
            sig = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        pass
    return (float(t0) + float(res.x[0]) * t_scale, float(np.exp(res.x[1])),
            abs(N) / np.sqrt(M), N / M, sig[0] * t_scale, sig[1])


def core_interval(event):
    """[t_lo, t_hi) spanned by an excess-power event's core clusters. The year is periodic and
    clusters of different tilings can be stored on either side of its join, so each is first
    moved by a whole period to lie nearest the seed cluster."""
    s = event.seed
    t_obs = s.level.nt * s.level.delta_t
    centre = 0.5 * (s.t_lo + s.t_hi)
    los, his = [], []
    for c in event.core:
        shift = t_obs * np.round((centre - 0.5 * (c.t_lo + c.t_hi)) / t_obs)
        los.append(c.t_lo + shift)
        his.append(c.t_hi + shift)
    return min(los), max(his)


def event_window(event, pad_min=1800.0):
    """The event's core interval padded by one seed bin and at least pad_min on either side:
    the onset precedes most of a long glitch's power."""
    lo, hi = core_interval(event)
    pad = max(event.seed.level.delta_t, pad_min)
    return lo - pad, hi + pad


def partition_windows(events, t_obs, pad_min=1800.0):
    """event_window for every event, clipped at the midpoint between its core and each
    neighbouring event's core. Without the clip, the window of a faint event next to a loud
    one reaches the loud glitch, both events are seeded with it, and it is subtracted
    twice while the faint one is never fitted."""
    n = len(events)
    cores = [core_interval(e) for e in events]
    wins = [event_window(e, pad_min) for e in events]
    if n < 2:
        return wins
    centres = np.array([(0.5 * (lo + hi)) % t_obs for lo, hi in cores])
    order = np.argsort(centres)
    out = list(wins)
    for r, i in enumerate(order):
        lo_i, hi_i = cores[i]
        c_i = 0.5 * (lo_i + hi_i)
        prev, nxt = cores[order[(r - 1) % n]], cores[order[(r + 1) % n]]
        # place the previous core just before event i and the next one just after it
        sp = t_obs * np.floor((c_i - 0.5 * (prev[0] + prev[1])) / t_obs)
        sn = t_obs * np.ceil((c_i - 0.5 * (nxt[0] + nxt[1])) / t_obs)
        hi_p, lo_n = prev[1] + sp, nxt[0] + sn
        left = min(lo_i, 0.5 * (hi_p + lo_i))
        right = max(hi_i, 0.5 * (hi_i + lo_n))
        out[i] = (max(wins[i][0], left), min(wins[i][1], right))
    return out


def followup(events, data_fd, mf, scan=None, refine_links=True):
    """A seed for every excess-power event: the best bank template whose onset lies in the
    event's window (partition_windows), refined continuously for both test masses of its
    arm; None where the seed repeats a louder one (dedupe). `scan` is mf.scan(data_fd),
    computed here if not given."""
    grid = mf.grid
    sc = mf.scan(data_fd) if scan is None else scan
    Y = np.asarray(wdm.whiten(data_fd, grid, mf.channels))
    n = sc.z.size
    seeds = []
    for e, (lo, hi) in zip(events, partition_windows(events, grid.t_obs)):
        j = np.arange(int(np.floor(lo / sc.dt)), int(np.ceil(hi / sc.dt))) % n
        jb = int(j[np.argmax(sc.absz[j])])
        tau_b, arm = mf.templates[int(sc.itpl[jb])]
        t_b = jb * sc.dt
        best = None
        for link in ((arm, PARTNER[arm]) if refine_links else (arm,)):
            t0, tau, rho, dv, s_t0, s_lt = refine(Y, grid, t_b, tau_b, link, mf.channels)
            if best is None or rho > best.rho:
                best = Seed(t0=t0 % grid.t_obs, dv=dv, tau=tau, link=link, rho=rho,
                            z_bank=float(sc.z[jb]), tau_bank=tau_b,
                            sigma_t0=s_t0, sigma_logtau=s_lt)
        seeds.append(best)
    return dedupe(seeds, grid.t_obs)


def same_glitch(a, b, t_obs):
    """Two seeds of one glitch: the same arm, onsets within max(120 s, half the longer tau)."""
    if b.link not in (a.link, PARTNER[a.link]):
        return False
    dt = abs((a.t0 - b.t0 + 0.5 * t_obs) % t_obs - 0.5 * t_obs)
    return dt <= max(120.0, 0.5 * max(a.tau, b.tau))


def dedupe(seeds, t_obs):
    """Replace by None every seed that repeats a louder one of the same call. Windows are
    partitioned, but events whose cores overlap can still reach the same glitch (a triple
    within 25 min did, in an LPF-like year); subtracting it twice would plant its negative
    in the residual. The event left without a seed is searched again on the next pass."""
    out = list(seeds)
    order = sorted(range(len(seeds)), key=lambda i: -seeds[i].rho)
    for r, i in enumerate(order):
        if out[i] is None:
            continue
        for j in order[r + 1:]:
            if out[j] is not None and same_glitch(out[i], out[j], t_obs):
                out[j] = None
    return out

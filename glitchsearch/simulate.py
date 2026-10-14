"""Frequency-domain simulation of one year of equal-arm TDI-1 data: noise plus glitches.

Everything is generated directly on the one-sided grid of `grid.Grid`, up to its Nyquist
frequency and no further, which is what "working from the frequency domain" buys: a year
at LISA's native cadence would be 1.3e8 samples per channel, while the band that holds a
glitch's SNR needs 2.6e5 bins. The implied time series is periodic with period T_obs,
exactly as in the jaxglitches examples, so a glitch near the end of the year wraps round
to its start, and so does the search.

Glitch templates are `jaxglitches.clean_signal_f` (equal arms, TDI-1, the integrated n = 1
shapelet on test mass 12). The other five test masses follow from relabelling the
spacecraft, which is exact for equal arms (see `_LINK_MAP`). Nothing in the search depends
on that choice: the A + E pixel power it sums is invariant under all six, because the
relabelling acts on (A, E) as a rotation or a reflection and S_A = S_E
(`tests/test_search.py` checks it pixel by pixel).
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

import jaxglitches as jg

from . import noise as ns

jax.config.update("jax_enable_x64", True)

LINKS = ("12", "23", "31", "13", "32", "21")      # order of jaxglitches.INJ_POINTS

# (X, Y, Z) of a glitch on test mass `link` from those of a glitch on test mass 12,
# [X12, Y12, Z12 = 0], as (source, sign) per output channel. Relabelling the spacecraft
# cyclically, 1 -> 2 -> 3 -> 1, maps link 12 to 23 and the Michelson centred on 1 to the
# one centred on 2, so (X, Y, Z)[23] = (Z, X, Y)[12]. Swapping 2 <-> 3 maps 12 to 13 and
# exchanges the two arms of every Michelson, flipping its sign, so
# (X, Y, Z)[13] = -(X, Z, Y)[12]. The remaining three follow by composition. Both
# relabellings are symmetries only for equal arms.
_LINK_MAP = {
    "12": ((0, +1), (1, +1), (2, +1)),
    "23": ((2, +1), (0, +1), (1, +1)),
    "31": ((1, +1), (2, +1), (0, +1)),
    "13": ((0, -1), (2, -1), (1, -1)),
    "21": ((1, -1), (0, -1), (2, -1)),
    "32": ((2, -1), (1, -1), (0, -1)),
}
_S2, _S3, _S6 = np.sqrt(2.0), np.sqrt(3.0), np.sqrt(6.0)
_XYZ_TO_AET = np.array([[-1 / _S2, 0.0, 1 / _S2],
                        [1 / _S6, -2 / _S6, 1 / _S6],
                        [1 / _S3, 1 / _S3, 1 / _S3]])


def _mixing(link):
    M = np.zeros((3, 3))
    for out, (src, sign) in enumerate(_LINK_MAP[link]):
        M[out, src] = sign
    return _XYZ_TO_AET @ M


MIXING = jnp.asarray(np.stack([_mixing(link) for link in LINKS]))   # [A,E,T] = MIXING[i] @ [X,Y,Z]_12


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def noise_fd(key, grid):
    """One Gaussian noise realisation [A, E, T], shape (F, 3), with E|n_k|^2 equal to the
    per-bin variance of `grid`; the DC bin is zero (jaxglitches examples' `sample_noise_fd`)."""
    return ns.sample_noise_fd(key, jnp.asarray(grid.bin_variance))


# ---------------------------------------------------------------------------
# Glitches
# ---------------------------------------------------------------------------

@jax.jit
def _xyz12(params, freq):
    return jg.clean_signal_f(params, freq, tdi=1, basis="XYZ")


def glitch_fd(params, grid, link: str = "12"):
    """Equal-arm TDI-1 response [A, E, T], shape (F, 3), of one glitch.

    params : [t0 (s), Deltav (m/s), tau (s)], jaxglitches conventions (onset, velocity
             kick with its sign as polarity, decay time).
    link   : injection test mass, one of LINKS.
    """
    xyz = _xyz12(jnp.asarray(params, dtype=float), jnp.asarray(grid.freq))
    return xyz @ MIXING[LINKS.index(link)].T


@partial(jax.jit, static_argnames=("chunk",))
def _inject(params, link_idx, freq, chunk):
    p = params.reshape(-1, chunk, 3)
    li = link_idx.reshape(-1, chunk)

    def body(acc, xs):
        pc, lc = xs
        xyz = jax.vmap(lambda q: jg.clean_signal_f(q, freq, tdi=1, basis="XYZ"))(pc)
        return acc + jnp.einsum("cfj,cij->fi", xyz, MIXING[lc]), None

    return lax.scan(body, jnp.zeros((freq.size, 3), jnp.complex128), (p, li))[0]


def inject(params, link_idx, grid, chunk: int = 16):
    """Sum of many glitches [A, E, T], shape (F, 3).

    params   : (n, 3) array of [t0, Deltav, tau].
    link_idx : (n,) integer index into LINKS.
    Glitches are evaluated `chunk` at a time, so a year of LPF-like events (~1.6e3) never
    holds more than `chunk` templates in memory.
    """
    params = np.atleast_2d(np.asarray(params, dtype=float))
    link_idx = np.asarray(link_idx, dtype=int).reshape(-1)
    n = params.shape[0]
    if n == 0:
        return jnp.zeros((grid.n_freq, 3), jnp.complex128)
    pad = (-n) % chunk          # padding glitches have Deltav = 0, hence no signal
    params = np.concatenate([params, np.tile([0.0, 0.0, 1.0], (pad, 1))])
    link_idx = np.concatenate([link_idx, np.zeros(pad, int)])
    return _inject(jnp.asarray(params), jnp.asarray(link_idx), jnp.asarray(grid.freq), chunk)


# ---------------------------------------------------------------------------
# Signal-to-noise ratios
# ---------------------------------------------------------------------------

def snr(h_fd, grid, channels=(0, 1, 2)):
    """Optimal SNR sqrt((h|h)) over the chosen channels (default A, E, T), with the inner
    product of jaxglitches.likelihood."""
    ch = list(channels)
    return float(jg.snr(jnp.asarray(h_fd)[:, ch], jnp.asarray(grid.bin_variance)[:, ch]))


@jax.jit
def _snr_unit(taus, freq, var):
    """SNR of a unit-Deltav glitch for each tau. It is independent of t0 (a phase) and of
    the test mass (see module docstring), so rho = |Deltav| * R(tau) for every event."""
    def one(tau):
        h = jg.clean_signal_f(jnp.array([0.0, 1.0, tau]), freq, tdi=1)
        return jnp.sqrt(2.0 * jnp.sum((jnp.abs(h[1:]) ** 2) / var[1:]))
    return lax.map(one, taus)


def snr_of(params, grid):
    """Optimal SNR (A + E + T) of each glitch in an (n, 3) parameter array."""
    params = np.atleast_2d(np.asarray(params, dtype=float))
    R = np.asarray(_snr_unit(jnp.asarray(params[:, 2]), jnp.asarray(grid.freq),
                             jnp.asarray(grid.bin_variance)))
    return np.abs(params[:, 1]) * R


def with_snr(params, rho, grid):
    """Copy of an (n, 3) parameter array with each Deltav rescaled, sign kept, so that the
    glitch has optimal SNR rho (scalar or (n,))."""
    params = np.array(np.atleast_2d(params), dtype=float)
    unit = params.copy()
    unit[:, 1] = 1.0
    params[:, 1] = np.where(params[:, 1] < 0, -1.0, 1.0) * np.asarray(rho) / snr_of(unit, grid)
    return params


# ---------------------------------------------------------------------------
# A year of LPF-like glitches
# ---------------------------------------------------------------------------

def lpf_catalogue(key, grid, smooth: bool = False, max_deltav=3e-11):
    """Poisson arrivals at the LPF ordinary-run rate (~4.3 per day) with (Deltav, tau)
    resampled from the LPF catalogue, via jaxglitches.run_catalog.

    smooth     : False (default) bootstraps the measured events and cannot leave their
                 support; True draws from a kernel-density estimate that reaches the prior's
                 |Deltav| = 1e-7 m/s, a factor 5 beyond anything LPF measured (jaxglitches
                 paper, Sec. "Population and priors").
    max_deltav : drop events with |Deltav| above this, in m/s. The default removes the two
                 catalogue rows the jaxglitches paper sets aside, 2.1e-8 and 6.5e-10 m/s,
                 whose amplitudes contradict their own LPF SNRs (the first by a factor
                 5.6e3); the loudest self-consistent event is 2.7e-11 m/s. The bootstrap
                 draws the 2.1e-8 row a few times a year, at rho ~ 3e5 on this grid, and
                 its wavelet sidelobes then clear threshold for days (README, "Limits").
                 None keeps every row.

    Returns (params (n, 3), link_idx (n,), rho (n,), n_dropped).
    """
    cat = jg.run_catalog(grid.t_obs, key=key, smooth=smooth)
    params = np.asarray(cat["params"], dtype=float).reshape(-1, 3)
    link_idx = np.asarray(cat["inj_idx"], dtype=int).reshape(-1)
    keep = (np.ones(len(params), bool) if max_deltav is None
            else np.abs(params[:, 1]) <= max_deltav)
    params, link_idx = params[keep], link_idx[keep]
    return params, link_idx, snr_of(params, grid), int((~keep).sum())

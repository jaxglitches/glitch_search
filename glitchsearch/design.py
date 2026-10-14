"""Predicted sensitivity of the search, from noiseless templates and the chi^2 laws.

For a glitch of unit A + E SNR, every tile of the bank captures a fraction lambda of rho^2
and has dof degrees of freedom. At a per-tile threshold eta(dof) the tile fires with
probability P(ncx2(dof, lambda rho^2) > eta), and the search with at least the largest of
these over the bank -- the tiles are nearly redundant for a given glitch, so the bound is
close. Averaging over where the glitch falls relative to the time bins gives the detection
probability as a function of (rho, tau), which `scripts/efficiency.py` checks against
injections.

Placements are absolute onset times shared by all tilings, spread quasi-randomly over the
coarsest bin (`placements`), because in a ladder a glitch has one arrival time, not one
phase per tiling.
"""
import numpy as np
import jax.numpy as jnp
from scipy import optimize, stats

from . import simulate, wdm

GOLDEN = 0.5 * (np.sqrt(5.0) - 1.0)


def placements(grid, n: int = 16, span=None):
    """n onset times around mid-year, spread quasi-randomly over `span` (default: the
    coarsest bin of the dyadic ladder, 17 h), so that every tiling sees ~uniform phases."""
    span = grid.t_obs / 512 if span is None else span
    return 0.5 * grid.t_obs + span * ((np.arange(1, n + 1) * GOLDEN) % 1.0)


def rho2_below(taus, f_cuts, f_min=1e-5, f_max=1.0, n=400_001):
    """Fraction of a glitch's rho^2 (A + E + T, continuous integral 4 int |h|^2 / S df over
    [f_min, f_max]) below each f_cut, the cumulative curves, and the share carried by T."""
    import jaxglitches as jg
    from . import noise as ns
    f = np.geomspace(f_min, f_max, n)
    S = np.stack(ns.psd_tdi1(jnp.asarray(f)), -1)
    frac, t_share, cum = [], [], []
    for tau in taus:
        H = np.asarray(jg.clean_signal_f(jnp.array([0.0, 1e-12, float(tau)]),
                                         jnp.asarray(f), tdi=1))
        dens = 4.0 * np.abs(H) ** 2 / S
        tot = dens.sum(1)
        c = np.concatenate([[0.0], np.cumsum(0.5 * (tot[1:] + tot[:-1]) * np.diff(f))])
        c /= c[-1]
        cum.append(c)
        frac.append(np.interp(f_cuts, f, c))
        per = np.trapezoid(dens, f, axis=0)
        t_share.append(per[2] / per.sum())
    return f, np.array(cum), np.array(frac), np.array(t_share)


def unit_power(grid, level, tau, t0, channels=(0, 1)):
    """Pixel power (nt, n_chan) of a noiseless glitch with onset t0, scaled to unit A + E SNR."""
    h = simulate.glitch_fd([t0, 1.0, float(tau)], grid)
    Y = wdm.whiten(h, grid, channels)
    Y = Y / jnp.sqrt(2.0 * jnp.sum(jnp.abs(Y) ** 2))
    return wdm.pixel_power(Y, level)


def captures(power, bands, ks, halo: int = 3):
    """Largest fraction of rho^2 captured near the glitch by a tile of each (k, [lo, hi])
    in the product of ks and bands (1-based inclusive channel ranges)."""
    nt = power.shape[0]
    npk = int(np.argmax(power.sum(1)))
    rows = power[np.arange(npk - halo, npk + halo + 1) % nt]
    cum = np.concatenate([np.zeros((rows.shape[0], 1)), np.cumsum(rows, 1)], 1)
    out = []
    for k in ks:
        for lo, hi in bands:
            band = cum[:, hi] - cum[:, lo - 1]
            out.append(max(band[i:i + k].sum() for i in range(rows.shape[0] - k + 1)))
    return np.array(out)


def level_response(grid, level, shapes, taus, t0s, channels=(0, 1)):
    """lambda (n_tau, n_t0, n_shapes) and dof (n_shapes,) of low-pass shapes on one tiling."""
    ks = sorted({s.k for s in shapes})
    his = sorted({s.m_hi for s in shapes})
    order = {(k, h): i for i, (k, h) in enumerate((k, h) for k in ks for h in his)}
    pick = [order[(s.k, s.m_hi)] for s in shapes]
    lam = np.empty((len(taus), len(t0s), len(shapes)))
    for i, tau in enumerate(taus):
        for j, t0 in enumerate(t0s):
            c = captures(unit_power(grid, level, tau, t0 - 2.0 * tau, channels),
                         [(1, h) for h in his], ks)
            lam[i, j] = c[pick]
    dof = np.array([len(channels) * s.k * s.m_hi for s in shapes], dtype=float)
    return lam, dof


def search_response(search, taus, t0s):
    """lambda (n_tau, n_t0, n_tiles) and dof (n_tiles,) over the search's whole bank."""
    parts = [level_response(search.grid, lev, bank, taus, t0s, search.channels)
             for lev, bank in zip(search.levels, search.banks)]
    return np.concatenate([p[0] for p in parts], axis=2), np.concatenate([p[1] for p in parts])


def lam_for(p_det, eta, dof):
    """Non-centrality at which ncx2(dof) exceeds eta with probability p_det."""
    return optimize.brentq(lambda x: stats.ncx2.sf(eta, dof, x) - p_det, 1e-9,
                           50.0 * (eta + 10.0))


def rho_for(p_det, lam, dof, log_p_thr):
    """SNR at which the best tile fires with probability p_det, median over placements.
    lam: (n_t0, n_tiles); dof: (n_tiles,)."""
    need = {}
    for d in np.unique(dof):
        need[d] = lam_for(p_det, stats.chi2.isf(np.exp(log_p_thr), d), d)
    req = np.array([need[d] for d in dof])
    with np.errstate(divide="ignore"):
        best = np.min(np.where(lam > 0, req[None, :] / lam, np.inf), axis=1)
    return float(np.median(np.sqrt(best)))


def detection_probability(rhos, lam, dof, log_p_thr):
    """P(best tile fires) against rho, averaged over placements. lam: (n_t0, n_tiles)."""
    rhos = np.asarray(rhos, dtype=float)
    eta = stats.chi2.isf(np.exp(log_p_thr), dof)
    p = stats.ncx2.sf(eta[None, None], dof[None, None], lam[None] * rhos[:, None, None] ** 2)
    return p.max(axis=2).mean(axis=1)

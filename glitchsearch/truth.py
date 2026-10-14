"""Ground truth for simulated data: what caused each event.

Matching events to injections by a time window fails in a realistic year: at the LPF rate
of ~4.4 glitches a day, a glitch usually has a neighbour within an hour, and a window wide
enough for the coarse tilings hands the neighbour's event to whichever glitch is louder.
Here the attribution is made where the event was made instead, in its seed tile. The
noiseless signal's energy in that tile, lambda_total, measured against the excess over
the noise mean the threshold demands, E_thr - dof, says how much of the event is signal;
each nearby glitch's own energy there, lambda_i, says which glitch it was.

    noise      lambda_total < 10% of the excess: noise made the event.
    assisted   10-50%, one glitch dominant (>= half of lambda_total): a glitch below the
               threshold on its own, carried over it by a noise fluctuation.
    glitch     >= 50%, one glitch dominant.
    pile-up    >= 10%, but no glitch holds half of lambda_total: several faint ones added up.

A glitch is found if it dominates a `glitch` or `assisted` event -- the same criterion an
isolated injection meets when an event lands on it (scripts/efficiency.py) -- and split if
it dominates more than one.
"""
import numpy as np
from scipy import stats

from . import simulate, wdm
from .simulate import LINKS

FOUND_KINDS = ("glitch", "assisted")


def attribute(events, scans_signal, params, links, search, log_p_thr, exclude=None):
    """Classify every event (see module docstring).

    scans_signal : search.scan(signal, robust=False) of the noiseless injected signal (or of
                   whatever noiseless signal the searched data held, e.g. a residual).
    exclude      : optional boolean mask of glitches that cannot be the cause (e.g. already
                   found and subtracted); an event whose energy is mostly theirs is then
                   `pile-up`, which for a residual means the imperfect subtraction.
    Returns a list of dicts: kind, dominant (index or -1), lam_total, lam_dom, excess.
    """
    grid = search.grid
    params = np.asarray(params, dtype=float)
    allowed = np.ones(len(params), bool) if exclude is None else ~np.asarray(exclude, bool)
    level_index = {lev.nt: i for i, lev in enumerate(search.levels)}
    out = []
    for e in events:
        s = e.seed
        lev = s.level
        sc = scans_signal[level_index[lev.nt]]
        lam_total = float(sc.energy[sc.shapes.index(s.shape), s.n0])
        excess = float(stats.chi2.isf(np.exp(log_p_thr), s.dof) - s.dof)
        rec = dict(kind="noise", dominant=-1, lam_total=lam_total, lam_dom=0.0, excess=excess)
        if lam_total >= 0.1 * excess:
            t_lo = (s.n0 - 0.5) * lev.delta_t
            t_hi = (s.n0 + s.shape.k - 0.5) * lev.delta_t
            d_lo = ((params[:, 0] - (t_lo - lev.delta_t) + 0.5 * grid.t_obs) % grid.t_obs
                    - 0.5 * grid.t_obs)
            reach = 5.0 * params[:, 2] + lev.delta_t * (s.shape.k + 2)
            cand = np.nonzero((d_lo > -reach) & (d_lo < t_hi - t_lo + 2 * lev.delta_t)
                              & allowed)[0]
            lam = np.array([_tile_energy(params[i], links[i], s, search) for i in cand])
            rec["kind"] = "pile-up"
            if cand.size:
                j = int(np.argmax(lam))
                rec.update(dominant=int(cand[j]), lam_dom=float(lam[j]))
                if lam[j] >= 0.5 * lam_total:
                    rec["kind"] = "glitch" if lam_total >= 0.5 * excess else "assisted"
        out.append(rec)
    return out


def _tile_energy(p, link_idx, seed, search):
    """Noiseless energy one glitch puts in the seed tile of an event."""
    lev, sh = seed.level, seed.shape
    Y = wdm.whiten(simulate.glitch_fd(p, search.grid, LINKS[int(link_idx)]), search.grid,
                   search.channels)
    power = wdm.pixel_power(Y, lev)
    rows = (seed.n0 + np.arange(sh.k)) % lev.nt
    return float(power[rows, :sh.m_hi].sum())


def summarise(attr, n_injections):
    """found (bool per injection), n_dominated (found-kind events per injection), and the
    number of events of each kind."""
    dom = np.array([a["dominant"] for a in attr if a["kind"] in FOUND_KINDS], dtype=int)
    n_dom = np.bincount(dom, minlength=n_injections) if dom.size else np.zeros(n_injections, int)
    kinds = {k: sum(a["kind"] == k for a in attr) for k in ("glitch", "assisted", "pile-up", "noise")}
    return n_dom > 0, n_dom, kinds

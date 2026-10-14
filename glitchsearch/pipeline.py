"""Search, follow up, subtract, search again: the glitch search as a loop.

The excess-power search finds glitches; the matched-filter follow-up (`mfilter.followup`)
turns each event into a single-glitch seed. Subtracting the seeds and searching the
residual is what the MBHB search of jaxmbhb lists as its next step (detect, subtract,
re-detect), and for glitches it repairs the two ways the single pass loses loud ones:

  * pairs.  Two glitches within ~30 min are one event, and the follow-up fits the louder;
    with it subtracted, the other is a glitch on its own in the residual.
  * sidelobes.  A glitch loud enough (rho ~ 1e5, the inconsistent row of the LPF
    catalogue) clears the threshold for days through its wavelet sidelobes, and the events
    there are pile-up, not glitches. Such a glitch has to go first, alone, before anything
    near it can be read, so the loop starts by fitting and subtracting, one at a time,
    every event whose excess power alone implies rho_ep > rho_loud, searching again after
    each.

After that, pass 0 is the ordinary search of the year and every event it makes is followed
up and subtracted. Pass p >= 1 searches the residual again, but keeps only events that
overlap a window of pass p - 1: a subtraction can only uncover a glitch where it subtracted
something, and restricting to those windows adds false alarms at the rate of the search
times the fraction of the year they cover (`Result.extra_far`), not a second full year's
worth.

A glitch must never be subtracted twice: its negative would be left in the residual and
found again as a "glitch", pass after pass. The follow-up guards against it twice over:
event windows are partitioned so that a faint event cannot reach a loud neighbour's glitch
(`mfilter.partition_windows`), and a seed that repeats a louder one of the same pass is
dropped (`mfilter.dedupe`) -- its event, if it is a glitch of its own, is found again on
the next pass with the louder one gone. Before both, one LPF-like year made 58 such ghost
events in passes 1-2; after, 2.

Only seeds the matched filter confirms, rho_seed >= rho_confirm = 5, are subtracted, and only
their windows searched again. A weaker seed means the event is noise, or not shaped like a
glitch: subtracting it would change nothing, and searching its window again would find the
same excess pass after pass. In an LPF-like year every event made by a glitch had
rho_seed >= 5.3, and the one noise event 3.6.
"""
from dataclasses import dataclass, field

import numpy as np

from . import mfilter, simulate
from .search import _circ_overlap


@dataclass
class Pass:
    events: list
    seeds: list                      # one per event; None where dedupe dropped a repeat
    subtracted: list                 # the confirmed seeds, rho >= rho_confirm
    windows: list                    # their events' windows, searched on the next pass


@dataclass
class Result:
    seeds: list                      # every seed, in the order it was subtracted
    loud: list                       # (event, seed) pairs of the loud-first stage
    passes: list = field(default_factory=list)
    residual: object = None          # (F, 3) data minus every seed
    extra_far: float = 0.0           # false alarms/yr added by passes >= 1 (window coverage)


def model_fd(seeds, grid):
    """Sum of the seeds' templates, (F, 3)."""
    if not seeds:
        return np.zeros((grid.n_freq, 3), complex)
    return np.asarray(simulate.inject(np.array([s.params for s in seeds]),
                                      np.array([s.link_idx for s in seeds]), grid))


def _coverage(windows, t_obs):
    """Fraction of the (periodic) year inside the union of windows."""
    if not windows:
        return 0.0
    n = 1 << 16
    mask = np.zeros(n, bool)
    for lo, hi in windows:
        j = np.arange(int(np.floor(lo / t_obs * n)), int(np.ceil(hi / t_obs * n))) % n
        mask[j] = True
    return float(mask.mean())


def run(data_fd, search, mf, log_p_thr, rho_loud=1e3, rho_confirm=5.0, max_passes=3,
        max_loud=20, far_per_year=1.0):
    """The loop (module docstring). Returns a Result; data_fd is left untouched."""
    grid = search.grid
    residual = np.array(data_fd)
    res = Result(seeds=[], loud=[])

    for _ in range(max_loud):                       # loud first, one at a time
        events, _ = search.run(residual, log_p_thr=log_p_thr)
        loud = [e for e in events if e.rho_ep > rho_loud]
        if not loud:
            break
        e = max(loud, key=lambda e: e.rho_ep)
        s = mfilter.followup([e], residual, mf)[0]
        residual = residual - model_fd([s], grid)
        res.loud.append((e, s))
        res.seeds.append(s)

    windows = None
    for p in range(max_passes):
        events, _ = search.run(residual, log_p_thr=log_p_thr)
        if windows is not None:
            events = [e for e in events
                      if any(_circ_overlap(c.t_lo, c.t_hi, lo, hi, grid.t_obs)
                             for c in e.core for lo, hi in windows)]
            res.extra_far += far_per_year * _coverage(windows, grid.t_obs)
        if not events:
            break
        seeds = mfilter.followup(events, residual, mf)          # None for duplicates
        ok = [s is not None and s.rho >= rho_confirm for s in seeds]
        kept = [s for s, k in zip(seeds, ok) if k]
        residual = residual - model_fd(kept, grid)
        windows = [mfilter.event_window(e) for e, k in zip(events, ok) if k]
        res.passes.append(Pass(events=events, seeds=seeds, subtracted=kept, windows=windows))
        res.seeds += kept
        if not windows:
            break
    res.residual = residual
    return res

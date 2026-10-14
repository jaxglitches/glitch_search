"""The matched filter, the follow-up, and the search-subtract loop."""
import numpy as np
import jax.random as jr

from glitchsearch import (Search, noise_fd, glitch_fd, inject, with_snr, snr, whiten, LINKS,
                          mfilter, pipeline)

TAUS = np.array([0.01, 113.0, 1010.0, 1.09e4])      # four decay times of the full bank


def test_matched_filter_is_unit_normal_under_noise(grid):
    """z = N / sqrt(M) is N(0, 1) at every onset, for short and long templates alike."""
    Y = np.asarray(whiten(noise_fd(jr.PRNGKey(21), grid), grid))
    for tau in (0.01, 1010.0):
        u = mfilter.unit_template(grid, tau)
        z = mfilter.correlate(Y, u / np.sqrt(mfilter.norm2(u)))
        assert abs(z.mean()) < 0.02 and abs(z.std() - 1.0) < 0.02


def test_test_masses_collapse_to_arms(grid):
    """In A + E the two test masses of an arm give one template; arms are 60 degrees apart."""
    u = {link: mfilter.unit_template(grid, 300.0, link) for link in LINKS}
    assert mfilter.match(u["12"], u["21"]) > 0.9999
    assert mfilter.match(u["23"], u["32"]) > 0.9999
    assert abs(mfilter.match(u["12"], u["23"]) - 0.5) < 0.01


def test_scan_finds_the_glitch_on_the_right_arm(grid):
    """Noiseless glitch: the bank recovers >= 95% of rho, at the onset, on the right arm."""
    p = with_snr([0.42 * grid.t_obs + 7.0, 1e-12, 900.0], 20.0, grid)[0]
    h = glitch_fd(p, grid, "13")
    mf = mfilter.MatchedFilter(grid, taus=TAUS)
    sc = mf.scan(h)
    j = int(np.argmax(sc.absz))
    assert sc.absz[j] > 0.95 * snr(h, grid, (0, 1))
    assert abs(j * sc.dt - p[0]) < 120.0
    assert mf.templates[sc.itpl[j]][1] == "31"                   # 13 lies on arm 31


def test_refine_recovers_noiseless_parameters(grid):
    """From a perturbed bank point the continuous maximisation returns the injected glitch."""
    p = with_snr([0.61 * grid.t_obs, -2e-12, 300.0], 30.0, grid)[0]
    h = glitch_fd(p, grid, "23")
    Y = np.asarray(whiten(h, grid))
    t0, tau, rho, dv, _, _ = mfilter.refine(Y, grid, p[0] + 25.0, 366.0, "23")
    assert abs(t0 - p[0]) < 0.05 and abs(tau / p[2] - 1) < 1e-4
    assert abs(dv / p[1] - 1) < 1e-4 and abs(rho / snr(h, grid, (0, 1)) - 1) < 1e-6


def test_followup_across_the_join(grid):
    """The fiducial glitch of the inference paper, 400 s into the periodic year: its event's
    clusters sit on both sides of the join, and the seed must still be found at the onset."""
    p = np.array([400.0, 1e-11, 300.0])
    d = noise_fd(jr.PRNGKey(1), grid) + glitch_fd(p, grid)
    events, _ = Search(grid).run(d, far_per_year=0.01)
    assert len(events) == 1
    lo, hi = mfilter.event_window(events[0])
    assert hi - lo < 2 * 86400.0
    seed = mfilter.followup(events, d, mfilter.MatchedFilter(grid, taus=TAUS))[0]
    dt = (seed.t0 - p[0] + 0.5 * grid.t_obs) % grid.t_obs - 0.5 * grid.t_obs
    assert abs(dt) < 30.0 and abs(seed.tau / p[2] - 1) < 0.2 and seed.link in ("12", "21")


def test_loop_resolves_a_close_pair(grid):
    """Two short glitches five minutes apart on different arms make one event; with the
    louder subtracted, the second pass finds the other."""
    t0 = 0.3 * grid.t_obs
    params = with_snr(np.array([[t0, 1e-12, 1.0], [t0 + 300.0, -1e-12, 1.0]]),
                      np.array([40.0, 20.0]), grid)
    d = noise_fd(jr.PRNGKey(8), grid) + inject(params, [0, 1], grid)
    search = Search(grid)
    res = pipeline.run(d, search, mfilter.MatchedFilter(grid, taus=TAUS),
                       search.threshold(0.01))
    assert len(res.passes[0].events) == 1
    assert len(res.passes) >= 2 and len(res.passes[1].events) == 1
    onsets = sorted(s.t0 for s in res.seeds)
    assert abs(onsets[0] - t0) < 60.0 and abs(onsets[1] - (t0 + 300.0)) < 60.0


def test_followup_windows_do_not_steal_a_loud_neighbour(grid):
    """A faint glitch 25 min before a loud one: two events, and unclipped windows would seed
    both with the loud glitch, which the loop then subtracts twice (found in an LPF-like
    year). Each event must be seeded with its own glitch."""
    t0 = 0.7 * grid.t_obs
    params = with_snr(np.array([[t0, 1e-12, 0.45], [t0 + 1500.0, 1e-12, 17.2]]),
                      np.array([16.0, 42.0]), grid)
    d = noise_fd(jr.PRNGKey(31), grid) + inject(params, [0, 2], grid)
    events, _ = Search(grid).run(d, far_per_year=0.01)
    assert len(events) == 2
    seeds = mfilter.followup(events, d, mfilter.MatchedFilter(grid, taus=TAUS))
    onsets = sorted(s.t0 for s in seeds)
    assert abs(onsets[0] - t0) < 120.0 and abs(onsets[1] - (t0 + 1500.0)) < 120.0

"""The excess-power statistic, its calibration, and recovery of injected glitches."""
import numpy as np
import jax.random as jr
import pytest
from scipy import stats

from glitchsearch import (Search, chi2_logsf, noise_fd, glitch_fd, whiten, pixel_power,
                          with_snr, LINKS)


def test_chi2_logsf_tail():
    """Equal to scipy where scipy is finite, continuous across the switch, finite beyond."""
    d = np.array([2.0, 8.0, 40.0, 300.0, 2948.0])
    for x in (d + 10 * np.sqrt(2 * d), d + 30 * np.sqrt(2 * d)):
        ref = stats.chi2.logsf(x, d)
        assert np.allclose(chi2_logsf(x, d), ref, rtol=1e-10, atol=0)
    # continued fraction against scipy in the overlap, just above the switch
    from glitchsearch.search import _log_q_cf
    a, z = 0.5 * d, 0.5 * (d + 40 * np.sqrt(2 * d))
    assert np.allclose(_log_q_cf(a, z), stats.chi2.logsf(2 * z, d), rtol=1e-10)
    far = chi2_logsf(np.array([1e4, 1e5, 1e6]), 8.0)
    assert np.all(np.isfinite(far)) and np.all(np.diff(far) < 0)


def test_statistic_is_link_independent(grid):
    """A + E pixel power is identical, pixel by pixel, for all six test masses."""
    p = [0.4 * grid.t_obs, 3e-12, 200.0]
    lev = grid.level(8192)
    ref = pixel_power(whiten(glitch_fd(p, grid, "12"), grid), lev)
    for link in LINKS[1:]:
        other = pixel_power(whiten(glitch_fd(p, grid, link), grid), lev)
        assert np.max(np.abs(other - ref)) < 1e-10 * ref.max()


def test_tile_energies_are_chi2_under_noise(grid):
    """Every tile's energy has the chi^2 mean and variance of its dof, and p-values are
    uniform: the fraction below 1e-2 is 1e-2 within binomial error."""
    search = Search(grid)
    scans = search.scan(noise_fd(jr.PRNGKey(11), grid))
    for sc in scans:
        z = (sc.energy - sc.dof[:, None]) / np.sqrt(2 * sc.dof[:, None])
        assert abs(z.mean()) < 0.05 and abs(z.std() - 1.0) < 0.05
    p = np.concatenate([np.exp(sc.logp).ravel() for sc in scans])
    assert abs(np.mean(p < 1e-2) - 1e-2) < 5 * np.sqrt(1e-2 / p.size) + 2e-4


@pytest.mark.parametrize("tau,t0_frac", [(1.0, 0.37), (300.0, 0.61), (3000.0, 0.83)])
def test_recovers_injected_glitch(grid, tau, t0_frac):
    """A rho = 25 glitch is found once, where it is, and nothing else is above 0.01/yr."""
    t0 = t0_frac * grid.t_obs
    p = with_snr([t0, 1e-12, tau], 25.0, grid)[0]
    d = noise_fd(jr.PRNGKey(int(tau) + 5), grid) + glitch_fd(p, grid)
    events, _ = Search(grid).run(d, far_per_year=0.01)
    assert len(events) == 1
    e = events[0]
    tol = max(e.seed.level.delta_t, 3 * tau)
    assert abs(e.t_hat - (t0 + tau)) < tol
    assert 15.0 < e.rho_ep < 30.0


def test_glitch_across_the_end_of_the_year(grid):
    """The transform is periodic: a glitch 100 s before the end is found, not split in two."""
    t0 = grid.t_obs - 100.0
    p = with_snr([t0, 1e-12, 50.0], 30.0, grid)[0]
    d = noise_fd(jr.PRNGKey(99), grid) + glitch_fd(p, grid)
    events, _ = Search(grid).run(d, far_per_year=0.01)
    assert len(events) == 1
    dt = (events[0].t_hat - t0 + 0.5 * grid.t_obs) % grid.t_obs - 0.5 * grid.t_obs
    assert abs(dt) < events[0].seed.level.delta_t + 150.0


def test_robust_mode_matches_known_psd(grid):
    """Median-renormalised channels find the same glitch, at nearly the same significance."""
    p = with_snr([0.45 * grid.t_obs, 1e-12, 100.0], 25.0, grid)[0]
    d = noise_fd(jr.PRNGKey(123), grid) + glitch_fd(p, grid)
    known, _ = Search(grid).run(d, far_per_year=0.01)
    robust, _ = Search(grid, robust=True).run(d, far_per_year=0.01)
    assert len(known) == len(robust) == 1
    assert abs(robust[0].log10p - known[0].log10p) < 0.05 * abs(known[0].log10p)


def test_truth_attribution(grid):
    """Two glitches two hours apart: each event is attributed to its own glitch, not to
    the louder neighbour a time window would hand it to."""
    from glitchsearch import inject, truth
    t0 = 0.5 * grid.t_obs
    params = with_snr(np.array([[t0, 1e-12, 1.0], [t0 + 7200.0, 1e-12, 10.0]]),
                      np.array([40.0, 15.0]), grid)
    signal = inject(params, [0, 3], grid)
    search = Search(grid)
    thr = search.threshold(0.01)
    events, _ = search.run(noise_fd(jr.PRNGKey(5), grid) + signal, log_p_thr=thr)
    attr = truth.attribute(events, search.scan(signal, robust=False), params, [0, 3], search, thr)
    found, n_dom, kinds = truth.summarise(attr, 2)
    assert found.all() and n_dom.max() == 1 and kinds["noise"] == 0


def test_far_curve_inverts():
    """threshold() inverts far(), inside the counted range and on the tail law beyond it."""
    from glitchsearch import FarCurve
    x = np.linspace(-20.0, -10.0, 30)
    counts = np.round(400 * 3e5 * np.exp(x))                  # rate0 = 3e5 /yr, 400 years
    curve = FarCurve(x, counts, years=400.0, n_trials=10 ** 6, record_years=1.0)
    for far in (1e-3, 0.3, 3.0, 10.0):
        assert abs(np.log(curve.far(curve.threshold(far)) / far)) < 0.05
    assert abs(curve.n_eff / 3e5 - 1.0) < 0.05
    with pytest.raises(ValueError):          # the grid stops at 13.6/yr: no extrapolating up
        curve.threshold(30.0)

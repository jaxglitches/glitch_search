"""The frequency-domain WDM kernel: exactness, normalisation, whitening."""
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import pytest
from wdm_transform import TimeSeries, WDM
from wdm_transform.backends import get_backend

from glitchsearch import forward, phi_window, whiten, noise_fd, glitch_fd, snr, LADDER


@pytest.mark.parametrize("nt,nf", [(8, 6), (64, 48), (256, 16)])
def test_kernel_matches_wdm_transform(nt, nf):
    """Interior channels from the one-sided spectrum equal wdm_transform's, to round-off."""
    x = np.random.default_rng(nt).standard_normal(nt * nf)
    ref = np.asarray(WDM.from_time_series(TimeSeries(jnp.asarray(x), dt=3.0,
                                                     backend=get_backend("jax")), nt=nt).coeffs)[0]
    mine = np.asarray(forward(jnp.asarray(np.fft.rfft(x)), nt, nf - 1))
    assert np.max(np.abs(ref[:, 1:nf] - mine)) < 1e-13 * np.max(np.abs(ref))


def test_window_is_normalised():
    for nt in (8, 512, 131072):
        assert abs(np.sum(phi_window(nt) ** 2) - 1.0) < 1e-12


def test_parseval_band_limited():
    """sum w^2 over the interior channels = 2 sum |Y|^2 for a spectrum inside their support."""
    nt, nf = 64, 32
    rng = np.random.default_rng(3)
    Y = np.zeros(nt * nf // 2 + 1, complex)
    Y[100:900] = rng.standard_normal(800) + 1j * rng.standard_normal(800)
    w = np.asarray(forward(jnp.asarray(Y), nt, nf - 1))
    assert abs((w ** 2).sum() / (2 * (np.abs(Y) ** 2).sum()) - 1.0) < 1e-12


def test_whitened_noise_pixels_are_unit_normal(grid):
    """Coloured SciRD noise, whitened in frequency, gives iid N(0, 1) pixels at every tiling,
    including the lowest channels where S_1 changes by an order of magnitude across one."""
    Y = whiten(noise_fd(jr.PRNGKey(7), grid), grid)
    for nt in LADDER:
        lev = grid.level(nt)
        w = np.asarray(forward(Y, nt, lev.n_chan))              # (2, nt, n_chan)
        n = w.size
        assert abs(w.mean()) < 5 / np.sqrt(n)
        assert abs(w.var() - 1.0) < 5 * np.sqrt(2.0 / n)
        low = w[:, :, :3]                                        # three lowest channels
        assert abs(low.var() - 1.0) < 5 * np.sqrt(2.0 / low.size)
        for lag in (np.mean(w[:, 1:] * w[:, :-1]), np.mean(w[:, :, 1:] * w[:, :, :-1])):
            assert abs(lag) < 5 / np.sqrt(n)


def test_glitch_power_is_rho_squared(grid):
    """At the coarsest tiling, where almost nothing lies below channel 1, the pixel power of a
    glitch sums to its optimal A + E SNR squared."""
    h = glitch_fd([0.3 * grid.t_obs, 1e-12, 30.0], grid)
    lev = grid.level(LADDER[0])
    w = np.asarray(forward(whiten(h, grid), lev.nt, lev.n_chan))
    rho2 = snr(h, grid, channels=(0, 1)) ** 2
    assert abs((w ** 2).sum() / rho2 - 1.0) < 1e-3

"""WDM transform of whitened frequency-domain data, straight from the one-sided spectrum.

The kernel is that of `wdm_transform` (Necula et al. 2012; Cornish 2020), restricted to the
interior channels and fed the one-sided spectrum directly. For interior channel m the
transform only reads Fourier bins k = m nt/2 + l, l in [-nt/2, nt/2), so

    w_nm = sqrt(2) (-1)^{nm} Re|Im [ sum_l Y_{m nt/2 + l} phi_l e^{2 pi i l n / nt} ]

(Re for n + m even, Im for n + m odd), one inverse FFT of length nt per channel. Channel m
needs nothing above bin (m + 1) nt / 2: the high-frequency cut of the data and the top
channel of the tiling are the same statement. `tests/test_wdm.py` checks this kernel
against `wdm_transform.WDM.from_time_series` to round-off.

Normalisation. The spectrum is whitened *before* the transform,

    Y_k = d_k / sigma_k,    sigma_k^2 = (T_obs / 2) S_1(f_k)    (jaxglitches bin variance),

which makes the noise white in the time domain; the WDM basis is orthonormal, so the pixels
are then exactly independent N(0, 1) for Gaussian stationary noise however coloured S_1 is,
and a signal obeys sum_nm w_nm^2 = 2 sum_k |Y_k|^2 = rho^2 (Parseval). A pixel's w^2 is its
share of rho^2. Whitening pixel by pixel after the transform would instead assume S_1 flat
across each channel's support, which fails in the lowest channels where S_1 goes as f^-2.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

A_WINDOW = 1.0 / 3.0                 # Meyer-window roll-off, the wdm_transform default
CHI2_1_MEDIAN = 0.45493642311957283  # median of chi^2 with one degree of freedom


def phi_window(nt: int, a: float = A_WINDOW):
    """The length-nt phi window in FFT order, scaled so that sum phi^2 = 1."""
    u = np.abs(2.0 * np.fft.fftfreq(nt))
    b = 1.0 - 2.0 * a
    w = np.where(u > a, np.cos(0.5 * np.pi * (u - a) / b), 1.0)
    return np.where(u > a + b, 0.0, w) * np.sqrt(2.0 / nt)


@partial(jax.jit, static_argnames=("nt", "n_chan"))
def forward(Y, nt: int, n_chan: int):
    """WDM coefficients of channels 1 .. n_chan from a one-sided spectrum.

    Y      : (..., K) one-sided spectrum on the f_k = k / T_obs grid, K >= (n_chan + 1) nt / 2.
    Returns: (..., nt, n_chan), real.
    """
    half = nt // 2
    l = np.round(np.fft.fftfreq(nt) * nt).astype(int)
    m = np.arange(1, n_chan + 1)
    idx = m[:, None] * half + l[None, :]                          # (n_chan, nt)
    z = nt * jnp.fft.ifft(Y[..., idx] * jnp.asarray(phi_window(nt)), axis=-1)
    n = np.arange(nt)
    even = (n[None, :] + m[:, None]) % 2 == 0                     # C_nm = 1 (even), i (odd)
    sign = np.where((n[None, :] * m[:, None]) % 2 == 0, 1.0, -1.0)
    w = np.sqrt(2.0) * sign * jnp.where(even, z.real, z.imag)     # (..., n_chan, nt)
    return jnp.swapaxes(w, -1, -2)


def whiten(d_fd, grid, channels=(0, 1)):
    """Whitened one-sided spectra, shape (len(channels), F), DC zeroed. Default A and E."""
    ch = list(channels)
    Y = jnp.asarray(d_fd)[:, ch] / jnp.asarray(grid.sigma)[:, ch]
    return Y.at[0].set(0.0).T


def renormalise(w):
    """Divide each (TDI channel, WDM channel) row by a robust noise level: the median of w^2
    over time bins against that of chi^2_1. A glitch occupies a handful of the nt time bins
    and cannot move a median; a stationary line (a Galactic binary) or a mis-estimated PSD
    is absorbed exactly as a PSD re-estimate would absorb it. w: (..., nt, n_chan)."""
    scale = jnp.median(w ** 2, axis=-2, keepdims=True) / CHI2_1_MEDIAN
    return w / jnp.sqrt(scale)


def pixel_power(Y, level, robust: bool = False):
    """Pixel power summed over the whitened channels in Y (A + E by default), shape
    (nt, n_chan): chi^2 with len(Y) degrees of freedom per pixel under noise."""
    w = forward(Y, level.nt, level.n_chan)
    if robust:
        w = renormalise(w)
    return np.asarray(jnp.sum(w ** 2, axis=0))

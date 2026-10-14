"""The analysis grid: one year of equal-arm TDI-1 data, band-limited in frequency.

Everything in this package lives on the one-sided Fourier grid f_k = k / T_obs of a
record of N samples at spacing dt, in the conventions of jaxglitches: templates are
continuous Fourier transforms (units of s), and the noise enters through the per-bin
variance sigma_k^2 = (T_obs / 2) S_1(f_k) that `noise.psd_tdi1_array` returns and that
`jaxglitches.likelihood` divides by.

Why dt = 60 s
-------------
Generating the data in the frequency domain means choosing where to stop, and that choice
is the sampling interval: dt = 60 s stops at f_Nyq = 8.33 mHz. The cut is set by where a
glitch's *signal-to-noise ratio* ends, not where its power ends. Above ~3 mHz the
optical-metrology noise in TDI fractional frequency rises as f^4, while a glitch's TDI-1
response is at best flat (tau -> 0, knee far above the band), so its SNR density falls at
least as f^-4. Below 8.33 mHz every glitch keeps >= 98% of its rho^2 -- the shortest are
the worst case; anything with tau >~ 30 s keeps > 99.9% -- and the best excess-power tile
for any glitch tops out at 4-5 mHz. The 3 mHz band of the inference paper would cost the
shortest glitches 30% of rho^2. `scripts/design_tiling.py` measures all three statements.

Why N = 2^19
------------
T_obs = 2^19 x 60 s = 31 457 280 s = 364.1 d. A power of two lets every dyadic WDM tiling
divide the record exactly; truncating the record to fit a tiling would break the
periodicity the frequency-domain simulation has by construction.

Why a ladder of tilings
-----------------------
A WDM tiling with nt time bins has Delta_t = T_obs / nt and Delta_F = nt / (2 T_obs); its
interior channels m = 1 .. nf - 1 (nf = N / nt) sit at f_m = m Delta_F, and pixel n is
centred on t_n = n Delta_t. The best excess-power tile tracks the glitch's *whitened*
duration, which runs from minutes (tau -> 0, where the noise bandwidth sets it) to days
(tau ~ 1e4 s). No single nt covers that, so LADDER holds five, a factor 4 apart:

    nt        Delta_t    Delta_F     most sensitive to
    512       17.1 h       8.1 uHz   tau >~ 1.5e4 s
    2048       4.3 h      32.6 uHz   tau ~ 3e3 - 1.5e4 s
    8192      64 min     130   uHz   tau ~ 5e2 - 3e3 s
    32768     16 min     521   uHz   tau ~ 30 - 5e2 s
    131072     4 min    2083   uHz   tau <~ 30 s, the bulk of the LPF population

Together they hold the SNR needed for 50% detection at a false-alarm rate of 1/yr between
7.1 and 8.4 from tau = 0.1 s to 2.3e4 s (9.9 at 5e4 s), never more than 15% above the best
single tiling for that tau. The five scan 2.4 times the tiles of the finest level alone,
which raises the threshold by 0.4 dex, a 3% cost in SNR (6.9 -> 7.1 at short tau).
Scanning all nine dyadic levels narrows the 15% to 9%, at the price of four more tilings
to scan and threshold.

If one tiling has to be chosen, SINGLE = 8192 (Delta_t = 64 min, Delta_F = 130 uHz) is the
minimax choice over tau = 1 s .. 1e4 s: never worse than 52% above the best tiling for that
tau (the worst case is tau -> 0, rho_50 = 10.4 against 6.9). The finer 16384 (32 min) is
better below tau ~ 2e3 s but reaches 98% worse at 5e3 s.
"""
import numpy as np
import jax
import jax.numpy as jnp

from . import noise as ns

jax.config.update("jax_enable_x64", True)

DT_S = 60.0                                  # f_Nyq = 8.33 mHz
N_SAMPLES = 2 ** 19                          # T_obs = 364.1 d
LADDER = (512, 2048, 8192, 32768, 131072)    # WDM time-bin counts, coarse to fine
SINGLE = (8192,)                             # the minimax one-tiling alternative
YEAR_S = 365.25 * 86400.0                    # false-alarm rates are quoted per Julian year


class Level:
    """One WDM tiling of the record: nt time bins by nf channels, of which the interior
    ones, m = 1 .. nf - 1, are analysed (DC and Nyquist are rank-deficient)."""

    def __init__(self, nt: int, grid: "Grid"):
        if nt % 2 or grid.n % nt or (grid.n // nt) % 2:
            raise ValueError(f"nt = {nt} must be even and divide N = {grid.n} into an even nf")
        self.nt = int(nt)
        self.nf = grid.n // self.nt
        self.n_chan = self.nf - 1
        self.delta_t = grid.t_obs / self.nt
        self.delta_f = self.nt / (2.0 * grid.t_obs)
        self.f_chan = np.arange(1, self.nf) * self.delta_f     # centres of channels 1 .. nf-1
        self.t_bin = np.arange(self.nt) * self.delta_t          # centres of time bins

    def __repr__(self):
        return (f"Level(nt={self.nt}, nf={self.nf}, Delta_t={self.delta_t / 60:.1f} min, "
                f"Delta_F={self.delta_f * 1e6:.1f} uHz)")


class Grid:
    """One-sided Fourier grid of the record, and the noise per bin on it."""

    def __init__(self, dt: float = DT_S, n: int = N_SAMPLES):
        self.dt = float(dt)
        self.n = int(n)
        self.t_obs = self.n * self.dt
        self.freq = np.fft.rfftfreq(self.n, self.dt)
        self.f_safe = np.where(self.freq > 0, self.freq, 1.0)
        self.n_freq = self.freq.size
        self.years = self.t_obs / YEAR_S
        self._bin_variance = None

    @property
    def bin_variance(self):
        """(F, 3) per-bin noise variance [A, E, T] in s^2, i.e. (T_obs / 2) S_1(f_k) for the
        TDI-1 SciRD noise of the jaxglitches examples. The DC row is a placeholder."""
        if self._bin_variance is None:
            self._bin_variance = np.asarray(
                ns.psd_tdi1_array(jnp.asarray(self.f_safe), t_obs=self.t_obs))
        return self._bin_variance

    @property
    def sigma(self):
        return np.sqrt(self.bin_variance)

    def level(self, nt: int) -> Level:
        return Level(nt, self)

    def levels(self, ladder=LADDER):
        return [self.level(nt) for nt in ladder]

    def __repr__(self):
        return (f"Grid(dt={self.dt} s, N={self.n}, T_obs={self.t_obs / 86400:.2f} d, "
                f"f_Nyq={0.5 / self.dt * 1e3:.2f} mHz)")

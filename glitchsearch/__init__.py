"""glitchsearch -- an excess-power search for LISA glitches in the WDM plane.

One year of equal-arm TDI-1 data is simulated in the frequency domain (jaxglitches
templates, the SciRD noise of the jaxglitches examples), whitened, transformed to a ladder
of WDM tilings straight from the one-sided spectrum, and scanned along the time axis with a
small bank of chi^2 tiles. See README.md for the design and the numbers behind it.
"""
from .grid import Grid, Level, LADDER, SINGLE, DT_S, N_SAMPLES, YEAR_S
from .simulate import (LINKS, noise_fd, glitch_fd, inject, snr, snr_of, with_snr,
                       lpf_catalogue)
from .wdm import forward, whiten, pixel_power, phi_window
from .search import Search, Shape, Event, Cluster, bank_for, chi2_logsf
from .far import FarCurve, noise_event_counts, threshold_grid, load_calibration

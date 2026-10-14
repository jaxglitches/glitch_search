"""The contract with jaxglitches: link labels, channel basis, and the seed hand-off."""
import numpy as np
import jax.numpy as jnp
import jax.random as jr

import jaxglitches as jg

from glitchsearch import glitch_fd, LINKS, lpf_catalogue
from glitchsearch.mfilter import Seed


def test_links_follow_jaxglitches():
    """LINKS is the order of jaxglitches' injection points and of its (lisaorbits) link
    delays, the order of the global fit's ltt_<link> catalogue columns."""
    assert LINKS == tuple(f"{link}" for link in jg.LINKS)
    assert LINKS == tuple(point.removeprefix("tm_") for point in jg.INJ_POINTS)


def test_catalogue_links(grid):
    """lpf_catalogue labels each glitch with the test mass jaxglitches drew for it."""
    cat = jg.run_catalog(10 * 86400.0, key=jr.PRNGKey(3), smooth=False)
    assert all(point == f"tm_{LINKS[i]}" for point, i in zip(cat["inj_point"], cat["inj_idx"]))
    params, link_idx, rho, _ = lpf_catalogue(jr.PRNGKey(3), grid)
    assert len(params) == len(link_idx) == len(rho) > 0
    assert link_idx.min() >= 0 and link_idx.max() < len(LINKS)


def test_link12_is_jaxglitches_aet(grid):
    """On test mass 12, glitch_fd is jaxglitches' own TDI-1 A, E, T response."""
    p = [1e6, 2e-12, 300.0]
    ours = np.asarray(glitch_fd(p, grid, "12"))
    ref = np.asarray(jg.clean_signal_f(jnp.asarray(p), jnp.asarray(grid.freq), tdi=1,
                                       basis="AET"))
    scale = np.max(np.abs(ref))
    assert np.max(np.abs(ours - ref)) < 1e-12 * scale


def test_seed_source():
    """A seed hands jaxglitches' parameter names on, with an absolute onset."""
    seed = Seed(t0=400.0, dv=-1e-11, tau=300.0, link="21", rho=40.0, z_bank=39.0,
                tau_bank=290.0, sigma_t0=7.0, sigma_logtau=0.02)
    src = seed.source(time_origin=1e9)
    assert src == {"t0": 1e9 + 400.0, "Deltav": -1e-11, "tau": 300.0, "link": "21"}
    assert np.array_equal([src[k] for k in ("t0", "Deltav", "tau")],
                          seed.params + [1e9, 0.0, 0.0])

"""One year of LISA data, searched for glitches.

showcase  Noise plus nine glitches spanning tau = 0.5 s .. 3e4 s and rho = 6 .. 100, among
          them the fiducial glitch of the jaxglitches paper (t0 = 400 s, Deltav = 1e-11 m/s,
          tau = 300 s), whose footprint straddles the end of the periodic year. Prints what
          the search finds next to what was injected, and draws the year and the WDM plane
          around the fiducial glitch.
lpf       Noise plus a year of LPF-like glitches (Poisson arrivals at ~4.3 per day,
          (Deltav, tau) bootstrapped from the LPF catalogue, random test masses): ~1.6e3
          events, most too faint to find. Reports the fraction found against rho next to
          isolated injections (scripts/efficiency.py), and what drove every event: one
          glitch, several faint ones together, or noise (glitchsearch/truth.py).

    python scripts/run_search.py [showcase|lpf] [--far 1.0] [--robust] [--seed 1] [--keep-all]

--keep-all keeps the two LPF catalogue rows whose amplitudes contradict their own SNRs
(simulate.lpf_catalogue); with them in, one glitch at rho ~ 3e5 swallows days of data.
Uses results/far_calibration.npz for the threshold if it exists (scripts/calibrate_far.py),
the Bonferroni bound otherwise. Writes figures/search_<tag>.pdf and results/search_<tag>.npz,
with <tag> the scenario plus _robust / _keepall when those flags are given.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import jax.random as jr  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from glitchsearch import (Grid, Search, LINKS, noise_fd, inject, with_snr, snr_of,  # noqa: E402
                          lpf_catalogue, load_calibration, truth)
from glitchsearch.plotting import plt, save, C, FULL_IN, SERIES  # noqa: E402

RES, FIG = HERE.parent / "results", HERE.parent / "figures"
DAY = 86400.0

# (t0 [s] or fraction of the year, tau [s], rho or None to keep Deltav, Deltav [m/s], link)
SHOWCASE = [
    (400.0, 300.0, None, 1e-11, "12"),        # the jaxglitches paper's fiducial glitch
    (0.11, 0.5, 8.0, 1e-12, "23"),
    (0.22, 5.0, 6.0, -1e-12, "31"),
    (0.33, 30.0, 12.0, 1e-12, "13"),
    (0.44, 100.0, 7.0, -1e-12, "32"),
    (0.55, 1000.0, 10.0, 1e-12, "21"),
    (0.66, 3000.0, 20.0, -1e-12, "12"),
    (0.77, 1e4, 15.0, 1e-12, "23"),
    (0.88, 3e4, 100.0, -1e-12, "31"),
]


def circ_dt(a, b, period):
    return (a - b + 0.5 * period) % period - 0.5 * period


def search_far(search, thr):
    return 10.0 ** search.log10_far_of(thr)


def search_with_truth(grid, search, thr, params, links, noise_key):
    """Inject, search, and attribute every event (glitchsearch/truth.py)."""
    signal = inject(params, links, grid)
    t1 = time.time()
    events, scans = search.run(noise_fd(noise_key, grid) + signal, log_p_thr=thr)
    t_search = time.time() - t1
    attr = truth.attribute(events, search.scan(signal, robust=False), params, links, search, thr)
    found, n_dom, kinds = truth.summarise(attr, len(params))
    print(f"search: {t_search:.2f} s, {len(events)} events above log10 p = {thr / np.log(10):.2f}")
    print(f"  made by one glitch: {kinds['glitch']}; by a sub-threshold glitch plus noise: "
          f"{kinds['assisted']}; by several faint glitches: {kinds['pile-up']}; by noise: "
          f"{kinds['noise']} (expected from noise alone: {grid.years * search_far(search, thr):.1f})")
    return events, scans, attr, found, n_dom


def run_showcase(grid, search, thr, seed, tag):
    params, links = [], []
    for t0, tau, rho, dv, link in SHOWCASE:
        t0 = t0 * grid.t_obs if t0 < 1 else t0
        p = np.array([[t0, dv, tau]])
        params.append(with_snr(p, rho, grid)[0] if rho is not None else p[0])
        links.append(LINKS.index(link))
    params, links = np.array(params), np.array(links)
    rho = snr_of(params, grid)
    events, scans, attr, found, n_dom = search_with_truth(grid, search, thr, params, links,
                                                           jr.PRNGKey(seed))

    print(f"\n{'injected':^40} | {'recovered':^72}")
    print(f"{'t0 [d]':>9} {'tau [s]':>8} {'rho':>6} {'link':>5} | {'t_hat - t0':>10} {'-log10 p':>9} "
          f"{'log10 FAR':>10} {'rho_ep':>6} {'seed tiling':>12} {'band top':>10} {'levels':>8}")
    for i in range(len(params)):
        js = [j for j, a in enumerate(attr) if a["kind"] in truth.FOUND_KINDS and a["dominant"] == i]
        head = (f"{params[i, 0] / DAY:9.3f} {params[i, 2]:8.1f} {rho[i]:6.1f} "
                f"{LINKS[links[i]]:>5} | ")
        if not js:
            print(head + "missed")
            continue
        m = events[min(js, key=lambda j: events[j].logp)].summary()
        print(head + f"{circ_dt(m['t_hat'], params[i, 0], grid.t_obs):8.0f} s {-m['log10p']:9.1f} "
              f"{m['log10_far']:10.1f} {m['rho_ep']:6.1f} {m['delta_t'] / 60:9.0f} min "
              f"{m['f_hi'] * 1e3:7.3f} mHz {len(m['levels']):8d}")
    for j, a in enumerate(attr):
        if a["kind"] not in truth.FOUND_KINDS:
            m = events[j].summary()
            print(f"{a['kind']:>31} | t_hat = {m['t_hat'] / DAY:.3f} d, -log10 p = "
                  f"{-m['log10p']:.2f}, FAR = {m['far']:.2f}/yr, tiling {m['delta_t'] / 60:.0f} min")

    fig = plt.figure(figsize=(FULL_IN, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1])
    ax = fig.add_subplot(gs[0, :])
    n_disp = 1024                                   # display only: max over ~8.5 h blocks
    t_disp = (np.arange(n_disp) + 0.5) * grid.t_obs / n_disp / DAY
    for i, (lev, s) in enumerate(zip(search.levels, search.significance(scans))):
        blk = s.reshape(n_disp, -1).max(1) if lev.nt >= n_disp else np.repeat(s, n_disp // lev.nt)
        ax.plot(t_disp, blk, color=SERIES[i], lw=0.8, ls=["-", "--", "-.", ":", "-"][i],
                label=f"$\\Delta t$ = {lev.delta_t / 60:.0f} min")
    ax.axhline(-thr / np.log(10), color=C["black"], lw=0.7, ls="--",
               label="threshold, 1 false alarm/yr")
    for i in range(len(params)):
        ax.axvline(params[i, 0] / DAY, color=C["grey"], lw=0.5, ls=":")
        ax.text(params[i, 0] / DAY, 1.02, f"$\\tau$={params[i, 2]:g} s\n$\\rho$={rho[i]:.0f}",
                transform=ax.get_xaxis_transform(), fontsize=5.5, ha="center", va="bottom")
    ax.set_yscale("log")
    ax.set(xlabel="time [d]", ylabel=r"most significant tile, $-\log_{10} p$",
           xlim=(-2, grid.t_obs / DAY + 2), ylim=(1.0, None))
    ax.legend(loc="upper left", ncols=3, bbox_to_anchor=(0.09, 0.98),
              title="max over 8.5 h blocks, per tiling", title_fontsize=6)

    fid = int(np.argmin(np.abs(params[:, 0] - 400.0)))
    js = [j for j, a in enumerate(attr) if a["kind"] in truth.FOUND_KINDS and a["dominant"] == fid]
    seed_tile = events[min(js, key=lambda j: events[j].logp)].seed if js else None
    for col, nt in enumerate((131072, 32768, 8192)):
        li = [lev.nt for lev in search.levels].index(nt)
        lev, sc = search.levels[li], scans[li]
        n_show = int(np.ceil(6 * 3600 / lev.delta_t)) + 2
        rows = np.arange(-n_show, n_show + 1)
        m_show = min(lev.n_chan, int(np.ceil(4e-3 / lev.delta_f)))
        img = sc.power[rows % lev.nt, :m_show].T
        axi = fig.add_subplot(gs[1, col])
        extent = [(rows[0] - 0.5) * lev.delta_t / 3600, (rows[-1] + 0.5) * lev.delta_t / 3600,
                  0.5 * lev.delta_f * 1e3, (m_show + 0.5) * lev.delta_f * 1e3]
        im = axi.imshow(img, origin="lower", aspect="auto", extent=extent, cmap="Blues",
                        vmin=0, vmax=max(10.0, np.percentile(img, 99.5)))
        if seed_tile is not None and seed_tile.level.nt == nt:
            sh, n0 = seed_tile.shape, seed_tile.n0
            n0 = n0 - lev.nt if n0 > lev.nt // 2 else n0
            axi.add_patch(Rectangle(((n0 - 0.5) * lev.delta_t / 3600, 0.5 * lev.delta_f * 1e3),
                                    sh.k * lev.delta_t / 3600, sh.m_hi * lev.delta_f * 1e3,
                                    fill=False, ec=C["red"], lw=1.2, label="seed tile"))
            axi.legend(loc="upper right", fontsize=6)
        axi.axvline(400 / 3600, color=C["orange"], lw=0.8, ls="--")
        axi.set(xlabel="time from the start of the year [h]",
                ylabel="frequency [mHz]" if col == 0 else None,
                title=f"$\\Delta t$ = {lev.delta_t / 60:.0f} min, $\\Delta F$ = "
                      f"{lev.delta_f * 1e6:.0f} $\\mu$Hz")
        axi.grid(False)
        fig.colorbar(im, ax=axi, pad=0.01, label=r"$w_A^2 + w_E^2$" if col == 2 else None)
    save(fig, FIG / f"search_{tag}.pdf")
    return params, links, rho, events, attr, found


def run_lpf(grid, search, thr, seed, tag, keep_all=False):
    params, links, rho, n_drop = lpf_catalogue(jr.PRNGKey(seed), grid,
                                               max_deltav=None if keep_all else 3e-11)
    print(f"LPF catalogue: {len(params)} glitches ({n_drop} inconsistent rows dropped); "
          f"rho > 8: {np.sum(rho > 8)}, > 20: {np.sum(rho > 20)}, > 100: {np.sum(rho > 100)}, "
          f"max {rho.max():.3g}; tau > 53 s: {np.sum(params[:, 2] > 53)}")
    events, scans, attr, found, n_dom = search_with_truth(grid, search, thr, params, links,
                                                           jr.PRNGKey(seed + 1))
    print(f"glitches found: {found.sum()}; found as more than one event: {np.sum(n_dom > 1)}")

    ref_path = RES / "efficiency.npz"
    ref = dict(np.load(ref_path)) if ref_path.exists() else None
    edges = np.array([3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 100, np.inf])
    print(f"\n{'rho bin':>12} {'injected':>9} {'found':>7} {'fraction':>9} | isolated, tau = 1 s")
    frac = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (rho >= lo) & (rho < hi)
        f = found[sel].mean() if sel.any() else np.nan
        frac.append(f)
        iso = ""
        if ref is not None:
            s = (ref["tau"] == 1.0) & (ref["rho"] >= lo) & (ref["rho"] < hi)
            iso = f"{ref['found'][s].mean():.2f}" if s.any() else ""
        print(f"{lo:5.0f}-{hi:<6.0f} {sel.sum():9d} {found[sel].sum():7d} {f:9.2f} | {iso}")
    loud_missed = np.nonzero((rho > 12) & ~found)[0]
    if loud_missed.size:
        near = []
        for i in loud_missed:
            d = np.abs(circ_dt(params[:, 0], params[i, 0], grid.t_obs))
            d[i] = np.inf
            near.append(np.any((d < 1800) & (rho >= 0.5 * rho[i])))
        print(f"\nmissed with rho > 12: {loud_missed.size}, of which {int(np.sum(near))} have a "
              f"glitch at least half as loud within 30 min (an unresolved pair)")

    dom = [(j, a["dominant"]) for j, a in enumerate(attr) if a["kind"] in truth.FOUND_KINDS]
    dt = np.array([circ_dt(events[j].t_hat, params[i, 0], grid.t_obs) for j, i in dom])
    short = np.array([params[i, 2] < 53 for _, i in dom])
    print(f"\ntiming of events made by one glitch: median |t_hat - t0| = "
          f"{np.median(np.abs(dt[short])):.0f} s (tau < 53 s), "
          f"{np.median(np.abs(dt[~short])) if (~short).any() else np.nan:.0f} s (tau > 53 s)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_IN, 2.5))
    centres = np.sqrt(edges[:-1] * np.minimum(edges[1:], 300))
    ax1.semilogx(centres, frac, "o-", color=C["blue"], ms=3.5, label="LPF-like year")
    if ref is not None:
        rr = np.unique(ref["rho"])
        ax1.semilogx(rr, [ref["found"][(ref["tau"] == 1.0) & (ref["rho"] == r)].mean() for r in rr],
                     "s--", color=C["orange"], ms=3.5, label=r"isolated injections, $\tau$ = 1 s")
    ax1.set(xlabel=r"optimal SNR $\rho$", ylabel="fraction found", ylim=(-0.03, 1.03),
            title="(a) found against SNR, at 1 false alarm/yr")
    ax1.legend(loc="lower right")
    ax2.scatter(params[~found, 2], rho[~found], s=3, color=C["grey"], label="missed", zorder=1)
    ax2.scatter(params[found, 2], rho[found], s=4, color=C["blue"], marker="s", label="found",
                zorder=2)
    ax2.set(xscale="log", yscale="log", xlabel=r"decay time $\tau$ [s]",
            ylabel=r"optimal SNR $\rho$", title="(b) the LPF-like year", ylim=(0.1, None))
    ax2.legend(loc="upper right")
    save(fig, FIG / f"search_{tag}.pdf")
    return params, links, rho, events, attr, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="showcase", choices=("showcase", "lpf"))
    ap.add_argument("--far", type=float, default=1.0, help="false alarms per year")
    ap.add_argument("--robust", action="store_true", help="median-renormalised channels")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--keep-all", action="store_true",
                    help="lpf: keep the catalogue rows with inconsistent amplitudes")
    args = ap.parse_args()
    grid = Grid()
    search = Search(grid, robust=args.robust, calibration=load_calibration(RES, args.robust))
    thr = search.threshold(args.far)
    print(grid, search, sep="\n")
    print(f"threshold at FAR = {args.far}/yr: log10 p = {thr / np.log(10):.3f} "
          f"({'calibrated' if search.calibration else 'Bonferroni'})\n")
    tag = (args.scenario + ("_robust" if args.robust else "")
           + ("_keepall" if args.keep_all and args.scenario == "lpf" else ""))
    if args.scenario == "showcase":
        params, links, rho, events, attr, found = run_showcase(grid, search, thr, args.seed, tag)
    else:
        params, links, rho, events, attr, found = run_lpf(grid, search, thr, args.seed, tag,
                                                          args.keep_all)
    RES.mkdir(exist_ok=True)
    np.savez(RES / f"search_{tag}.npz", params=params, links=links, rho=rho,
             found=found, t_hat=np.array([e.t_hat for e in events]),
             log10p=np.array([e.log10p for e in events]),
             kind=np.array([a["kind"] for a in attr]),
             dominant=np.array([a["dominant"] for a in attr]))
    print(f"  wrote {RES / f'search_{tag}.npz'}")


if __name__ == "__main__":
    main()

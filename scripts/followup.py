"""The matched-filter follow-up and the search-subtract loop, on the years of run_search.py.

showcase  The nine glitches of run_search.py, on the same noise: the seed each event hands
          to parameter estimation, next to what was injected. Among them is the inference
          paper's fiducial glitch, whose heterodyned likelihood needs t0 to ~2.6e3 s.
lpf       The LPF-like year(s) of run_search.py, same seeds, run through the loop of
          glitchsearch/pipeline.py: glitches loud enough to swamp their neighbours first,
          then every event followed up and subtracted, then the residual searched again
          where something was subtracted. Reports how accurate the seeds are against the
          glitches that made the events, and what the later passes add: glitches found only
          once a neighbour had been subtracted.

    python scripts/followup.py showcase
    python scripts/followup.py lpf [--seeds 1 2 3 4 5] [--keep-all]
    python scripts/followup.py noise [--years 40] [--far 10]

noise     Noise-only years searched at a loose threshold (default 10 false alarms/yr): every
          event is noise, and the SNR of its follow-up seed shows what the matched filter
          makes of an excess that is not a glitch -- the basis of the confirmation threshold
          rho_seed >= 5 with which the loop decides what to subtract.

Pass 0 of the loop is the search of run_search.py, event for event. Uses the calibrated
thresholds of results/ (calibrate_far.py) and the cached bank results/mf_bank.npz. Writes
results/followup_<tag>.npz and figures/followup_<tag>.pdf. About 10 min per LPF-like year
on a CPU, nearly all of it in the continuous refinement of ~600 seeds.
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

from glitchsearch import (Grid, Search, LINKS, noise_fd, inject, with_snr, snr_of,  # noqa: E402
                          lpf_catalogue, load_calibration, truth, mfilter, pipeline)
from glitchsearch.plotting import plt, save, C, FULL_IN, SERIES, MARKERS  # noqa: E402
from run_search import SHOWCASE, circ_dt  # noqa: E402

RES, FIG = HERE.parent / "results", HERE.parent / "figures"
DAY = 86400.0
HETERODYNE_S = 2.6e3        # |t0 - t_ref| the inference paper's binned likelihood tolerates


def showcase_params(grid):
    params, links = [], []
    for t0, tau, rho, dv, link in SHOWCASE:
        t0 = t0 * grid.t_obs if t0 < 1 else t0
        p = np.array([[t0, dv, tau]])
        params.append(with_snr(p, rho, grid)[0] if rho is not None else p[0])
        links.append(LINKS.index(link))
    return np.array(params), np.array(links)


def attribute_run(res, signal, params, links, search, thr):
    """Attribute every event of every stage to the truth. Pass-0 (and loud-first) events
    against the injected signal; pass-p events against the noiseless residual left by the
    subtractions before it, with glitches already found excluded as causes. Returns one
    record per seed: stage, kind, dominant glitch."""
    grid = search.grid
    found = np.zeros(len(params), bool)          # detected: dominates an event
    fitted = np.zeros(len(params), bool)         # ... and its seed was subtracted
    out = []
    subtracted = []
    for e, s in res.loud:                                        # loud first
        a = truth.attribute([e], search.scan(signal - pipeline.model_fd(subtracted, grid),
                                             robust=False), params, links, search, thr,
                            exclude=fitted)[0]
        out.append(dict(stage=-1, kind=a["kind"], dominant=a["dominant"], seed=s))
        if a["kind"] in truth.FOUND_KINDS:
            found[a["dominant"]] = fitted[a["dominant"]] = True
        subtracted.append(s)
    for p, ps in enumerate(res.passes):
        noiseless = signal - pipeline.model_fd(subtracted, grid)
        attr = truth.attribute(ps.events, search.scan(noiseless, robust=False), params, links,
                               search, thr, exclude=fitted if p > 0 else None)
        for a, sd in zip(attr, ps.seeds):
            out.append(dict(stage=p, kind=a["kind"], dominant=a["dominant"], seed=sd))
            if a["kind"] in truth.FOUND_KINDS:
                found[a["dominant"]] = True
                fitted[a["dominant"]] |= any(sd is x for x in ps.subtracted)
        subtracted += ps.subtracted
    return out, found


def seed_errors(recs, params, links, grid):
    """Seed against the glitch that made the event, for events of the found kinds."""
    rows = []
    for r in recs:
        if r["kind"] not in truth.FOUND_KINDS or r["seed"] is None:
            continue
        i, s = r["dominant"], r["seed"]
        rows.append((i, r["stage"], circ_dt(s.t0, params[i, 0], grid.t_obs), s.tau / params[i, 2],
                     s.dv / params[i, 1], s.link == LINKS[links[i]],
                     s.link in (LINKS[links[i]], mfilter.PARTNER[LINKS[links[i]]]), s.rho,
                     s.sigma_t0))
    names = ("glitch", "stage", "dt0", "tau_ratio", "dv_ratio", "link_exact", "arm_ok",
             "rho_seed", "sigma_t0")
    return {n: np.array([row[k] for row in rows]) for k, n in enumerate(names)}


def run_showcase(grid, search, mf, thr, seed):
    params, links = showcase_params(grid)
    rho = snr_of(params, grid)
    signal = np.asarray(inject(params, links, grid))
    data = np.asarray(noise_fd(jr.PRNGKey(seed), grid)) + signal
    t1 = time.time()
    res = pipeline.run(data, search, mf, thr)
    print(f"loop: {time.time() - t1:.1f} s, {len(res.passes)} passes, "
          f"{sum(len(p.events) for p in res.passes)} events, {len(res.loud)} loud-first")
    recs, found = attribute_run(res, signal, params, links, search, thr)
    err = seed_errors(recs, params, links, grid)
    print(f"\n{'injected':^42} | {'seed':^70}")
    print(f"{'t0 [d]':>9} {'tau [s]':>9} {'rho':>6} {'link':>5} | {'t0 err [s]':>10} {'sigma':>7} "
          f"{'tau':>9} {'Dv/Dv':>6} {'link':>5} {'rho':>6} {'pass':>5}")
    for i in range(len(params)):
        k = np.nonzero(err["glitch"] == i)[0] if err["glitch"].size else []
        head = f"{params[i, 0] / DAY:9.4f} {params[i, 2]:9.1f} {rho[i]:6.1f} {LINKS[links[i]]:>5} | "
        if len(k) == 0:
            print(head + "no seed")
            continue
        k = k[0]
        s = [r["seed"] for r in recs if r["kind"] in truth.FOUND_KINDS and r["dominant"] == i
             and r["seed"] is not None][0]
        print(head + f"{err['dt0'][k]:10.1f} {err['sigma_t0'][k]:7.1f} {s.tau:9.3g} "
              f"{err['dv_ratio'][k]:6.3f} {s.link:>5} {s.rho:6.1f} {err['stage'][k]:5d}")
    others = [r for r in recs if r["kind"] not in truth.FOUND_KINDS and r["seed"] is not None]
    for r in others:
        s = r["seed"]
        print(f"{r['kind']:>32} | seed at t0 = {s.t0 / DAY:.3f} d, rho = {s.rho:.1f}, "
              f"tau = {s.tau:.3g} s")
    return dict(params=params, links=links, rho=rho, found=found, **{f"err_{k}": v for k, v in err.items()})


def run_lpf(grid, search, mf, thr, seeds, keep_all):
    agg = {k: [] for k in ("rho", "tau", "found0", "found", "pair")}
    errs = []
    kinds = {}
    extra_far, n_loud_first, t_all = [], 0, time.time()
    for seed in seeds:
        params, links, rho, n_drop = lpf_catalogue(jr.PRNGKey(seed), grid,
                                                   max_deltav=None if keep_all else 3e-11)
        signal = np.asarray(inject(params, links, grid))
        data = np.asarray(noise_fd(jr.PRNGKey(seed + 1), grid)) + signal
        t1 = time.time()
        res = pipeline.run(data, search, mf, thr)
        recs, found = attribute_run(res, signal, params, links, search, thr)
        found0 = np.zeros(len(params), bool)
        for r in recs:
            if r["stage"] == 0 and r["kind"] in truth.FOUND_KINDS:
                found0[r["dominant"]] = True
        for r in recs:
            key = ("loud-first" if r["stage"] < 0 else f"pass {r['stage']}",
                   r["kind"] + ("" if r["seed"] is not None else ", duplicate seed dropped"))
            kinds[key] = kinds.get(key, 0) + 1
        pair = np.zeros(len(params), bool)
        for i in np.nonzero(rho > 12)[0]:
            d = np.abs(circ_dt(params[:, 0], params[i, 0], grid.t_obs))
            d[i] = np.inf
            pair[i] = np.any((d < 1800) & (rho >= 0.5 * rho[i]))
        for k, v in (("rho", rho), ("tau", params[:, 2]), ("found0", found0), ("found", found),
                     ("pair", pair)):
            agg[k].append(v)
        e = seed_errors(recs, params, links, grid)
        e["tau_true"] = params[e["glitch"].astype(int), 2] if e["glitch"].size else np.array([])
        e["rho_true"] = rho[e["glitch"].astype(int)] if e["glitch"].size else np.array([])
        errs.append(e)
        extra_far.append(res.extra_far)
        n_loud_first += len(res.loud)
        print(f"seed {seed}: {len(params)} glitches ({n_drop} rows dropped), max rho {rho.max():.3g}; "
              f"loud-first {len(res.loud)}, passes {[len(p.events) for p in res.passes]}; "
              f"found {found0.sum()} -> {found.sum()}; extra FAR {res.extra_far:.3f}/yr; "
              f"{time.time() - t1:.0f} s", flush=True)
    agg = {k: np.concatenate(v) for k, v in agg.items()}
    err = {k: np.concatenate([e[k] for e in errs]) for k in errs[0]}
    print(f"\n{len(seeds)} year(s), {time.time() - t_all:.0f} s")
    print("events by stage and kind: " + ", ".join(f"{a} {b}: {n}" for (a, b), n in sorted(kinds.items())))
    loud = agg["rho"] > 12
    print(f"glitches with rho > 12: {loud.sum()}; missed after pass 0: {(loud & ~agg['found0']).sum()} "
          f"({(loud & ~agg['found0'] & agg['pair']).sum()} pairs); after the loop: "
          f"{(loud & ~agg['found']).sum()} ({(loud & ~agg['found'] & agg['pair']).sum()} pairs)")
    print(f"found in total: {agg['found0'].sum()} -> {agg['found'].sum()}; "
          f"extra false alarms from passes >= 1: {np.mean(extra_far):.3f}/yr")
    short = err["tau_true"] < 53.0
    for lab, sel in (("tau < 53 s", short), ("tau > 53 s", ~short)):
        if not sel.any():
            continue
        adt = np.abs(err["dt0"][sel])
        print(f"seeds, {lab}: n = {sel.sum()}, |t0 err| median {np.median(adt):.1f} s, 90% "
              f"{np.percentile(adt, 90):.1f} s, 99% {np.percentile(adt, 99):.1f} s, "
              f"> {HETERODYNE_S:.0f} s: {(adt > HETERODYNE_S).sum()}; Dv ratio median "
              f"{np.median(err['dv_ratio'][sel]):.3f} (16-84%: "
              f"{np.percentile(err['dv_ratio'][sel], 16):.3f}-{np.percentile(err['dv_ratio'][sel], 84):.3f}); "
              f"arm right {err['arm_ok'][sel].mean():.3f}, link right {err['link_exact'][sel].mean():.3f}")
    long_ = ~short
    if long_.any():
        lr = np.log10(err["tau_ratio"][long_])
        print(f"tau for tau > 53 s: log10(tau_seed / tau) median {np.median(lr):+.3f}, 16-84% "
              f"{np.percentile(lr, 16):+.3f}..{np.percentile(lr, 84):+.3f}")
    return dict(agg_rho=agg["rho"], agg_tau=agg["tau"], agg_found0=agg["found0"],
                agg_found=agg["found"], agg_pair=agg["pair"], extra_far=np.array(extra_far),
                n_loud_first=n_loud_first, **{f"err_{k}": v for k, v in err.items()})


def run_noise(grid, search, mf, n_years, far):
    thr = search.threshold(far)
    rhos = []
    for y in range(n_years):
        d = np.asarray(noise_fd(jr.PRNGKey(50_000 + y), grid))
        events, _ = search.run(d, log_p_thr=thr)
        rhos += [sd.rho for sd in mfilter.followup(events, d, mf) if sd is not None]
    rhos = np.array(rhos)
    print(f"{n_years} noise years at {far}/yr: {rhos.size} events; seed rho quantiles "
          f"5/50/95/99% = {np.round(np.percentile(rhos, [5, 50, 95, 99]), 2)}, max {rhos.max():.2f}; "
          f"fraction with rho_seed >= 5: {np.mean(rhos >= 5.0):.3f}")
    return dict(noise_seed_rho=rhos, far=far, years=n_years)


def figure(out, tag):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_IN, 2.6))
    tau, dt0, rho = out["err_tau_true"], np.abs(out["err_dt0"]), out["err_rho_true"]
    bins = [(0, 53, r"$\tau<53$ s"), (53, 1e3, r"$53\leq\tau<10^3$ s"),
            (1e3, np.inf, r"$\tau\geq10^3$ s")]
    for i, (lo, hi, lab) in enumerate(bins):
        sel = (tau >= lo) & (tau < hi)
        if not sel.any():
            continue
        ax1.scatter(rho[sel], np.maximum(dt0[sel], 0.1), s=5 if i == 0 else 12,
                    color=SERIES[i], marker=MARKERS[i], label=f"{lab} ({sel.sum()})",
                    zorder=2 + i, linewidths=0)
    ax1.axhline(HETERODYNE_S, color=C["black"], lw=0.8, ls="--")
    ax1.text(0.02, HETERODYNE_S * 1.3, "heterodyne tolerance", fontsize=6,
             transform=ax1.get_yaxis_transform())
    ax1.set(xscale="log", yscale="log", xlabel=r"optimal SNR $\rho$ of the glitch",
            ylabel=r"$|t_0^{\rm seed} - t_0|$ [s]", title="(a) onset of the seed",
            ylim=(0.07, 3e5))
    ax1.legend(loc="upper right", markerscale=1.5)
    sel = tau >= 53
    ax2.scatter(tau[sel], np.maximum(tau[sel] * out["err_tau_ratio"][sel], 12.0), s=12,
                color=C["green"], marker="^", linewidths=0,
                label=rf"seeds of glitches with $\tau\geq53$ s ({sel.sum()})")
    lim = np.array([30, 1e5])
    ax2.plot(lim, lim, color=C["grey"], ls=":", lw=0.9, label="seed = truth")
    ax2.set(xscale="log", yscale="log", xlabel=r"injected $\tau$ [s]", ylabel=r"seed $\tau$ [s]",
            title="(b) decay time of the seed", xlim=lim, ylim=(10, 2e5))
    ax2.legend(loc="upper left")
    save(fig, FIG / f"followup_{tag}.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", default="showcase", choices=("showcase", "lpf", "noise"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--years", type=int, default=40, help="noise: number of noise years")
    ap.add_argument("--far", type=float, default=10.0, help="noise: false alarms per year")
    ap.add_argument("--figure", action="store_true", help="lpf: redraw from the stored npz only")
    ap.add_argument("--keep-all", action="store_true",
                    help="lpf: keep the catalogue rows with inconsistent amplitudes")
    args = ap.parse_args()
    grid = Grid()
    search = Search(grid, calibration=load_calibration(RES))
    thr = search.threshold(1.0)
    mf = mfilter.MatchedFilter(grid, cache=RES / "mf_bank.npz")
    print(grid, search, mf, sep="\n")
    print(f"threshold at FAR = 1/yr: log10 p = {thr / np.log(10):.3f}\n")
    tag = args.scenario + ("_keepall" if args.keep_all and args.scenario == "lpf" else "")
    if args.figure:
        figure(dict(np.load(RES / f"followup_{tag}.npz")), tag)
        return
    if args.scenario == "showcase":
        out = run_showcase(grid, search, mf, thr, args.seeds[0])
    elif args.scenario == "noise":
        out = run_noise(grid, search, mf, args.years, args.far)
    else:
        out = run_lpf(grid, search, mf, thr, args.seeds, args.keep_all)
        figure(out, tag)
    RES.mkdir(exist_ok=True)
    np.savez(RES / f"followup_{tag}.npz", **out)
    print(f"  wrote {RES / f'followup_{tag}.npz'}")


if __name__ == "__main__":
    main()

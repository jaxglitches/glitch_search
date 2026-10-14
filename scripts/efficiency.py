"""Detection efficiency: injections into noise, against the analytic prediction.

Glitches of five decay times (1 s to 2e4 s) and SNRs from 4 to 15 are injected into
noise-only years, 40 per year at least ~3.6 days apart, on random test masses; the search
runs at FAR = 1/yr (calibrated threshold if results/far_calibration.npz exists, else
Bonferroni). An injection counts as found if an event's core interval overlaps
[t0 - 1 h, t0 + 5 tau + 1 h]. The measured efficiency is set against
`design.detection_probability`, which knows nothing about the injections: a disagreement
would mean the chi^2 bookkeeping or the clustering loses glitches the tiles see.

Also recorded: the timing residual t_hat - t0 and the ratio rho_ep / rho, and every event
that matches no injection, as an independent check of the false-alarm rate.

    python scripts/efficiency.py [--per-cell 40] [--robust]
    python scripts/efficiency.py --per-cell 60 --mf

Writes results/efficiency[_robust].npz and figures/efficiency[_robust].pdf. About 1 min.

--mf also runs, on the same noise and the same injections, the matched-filter bank of
glitchsearch/mfilter.py at its own calibrated 1/yr threshold (results/far_calibration_mf.npz,
from calibrate_far.py --mf), and follows every injection the excess-power search found up
with mfilter.followup, recording the seed (t0, Delta v, tau, test mass) against the truth.
The excess-power columns are the ones the run without --mf produces, draw for draw. Writes
results/efficiency_mf.npz and figures/efficiency_mf.pdf. About 40 min.
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

from glitchsearch import (Grid, Search, LINKS, FarCurve, noise_fd, inject, with_snr,  # noqa: E402
                          load_calibration, design, mfilter)
from glitchsearch.search import _circ_overlap  # noqa: E402
from glitchsearch.plotting import plt, save, C, FULL_IN, SERIES, MARKERS  # noqa: E402

RES, FIG = HERE.parent / "results", HERE.parent / "figures"
TAUS = np.array([1.0, 30.0, 300.0, 3000.0, 2e4])
RHOS = np.array([4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0])
PER_YEAR = 40
FAR = 1.0
PAD = 3600.0


def window(t0, tau):
    return t0 - PAD, t0 + 5.0 * tau + PAD


def figure(rec, rho_fine, pred, tag):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_IN, 2.5))
    for i, tau in enumerate(TAUS):
        sel = rec["tau"] == tau
        n = np.array([(sel & (rec["rho"] == r)).sum() for r in RHOS])
        k = np.array([rec["found"][sel & (rec["rho"] == r)].sum() for r in RHOS])
        eff = k / n
        err = np.sqrt(np.clip(eff * (1 - eff), 1.0 / n, None) / n)
        ax1.errorbar(RHOS + 0.08 * (i - 2), eff, err, fmt=MARKERS[i], ms=3.5, color=SERIES[i],
                     lw=0.8, capsize=0, label=f"$\\tau$ = {tau:g} s")
        ax1.plot(rho_fine, pred[i], color=SERIES[i], lw=0.9)
    ax1.set(xlabel=r"injected SNR $\rho$", ylabel="fraction found",
            title="(a) efficiency at 1 false alarm/yr: injections (points), prediction (lines)",
            ylim=(-0.03, 1.03))
    ax1.legend(loc="lower right")
    q = np.array([np.percentile(rec["dt"][(rec["tau"] == tau) & rec["found"] & (rec["rho"] >= 8)],
                                [5, 50, 95]) for tau in TAUS])
    ax2.plot(TAUS, TAUS, color=C["grey"], ls=":", lw=0.9, label=r"$\hat t - t_0 = \tau$")
    ax2.errorbar(TAUS, q[:, 1], [q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], fmt="o", ms=4,
                 color=C["blue"], lw=1.0, capsize=2, label="median, 5-95% range")
    ax2.set_xscale("log")
    ax2.set_yscale("symlog", linthresh=100.0)
    ax2.set(xlabel=r"decay time $\tau$ [s]", ylabel=r"$\hat t - t_0$ [s]",
            title=r"(b) timing offset of glitches found at $\rho \geq 8$")
    ax2.legend(loc="upper left")
    save(fig, FIG / f"efficiency{tag}.pdf")


def circ(d, period):
    return (d + 0.5 * period) % period - 0.5 * period


def mf_columns(rec_mf, mf_events, mf_used, seeds, grid, t0, params, links, batch):
    """Append the matched-filter and follow-up records of one year (see --mf)."""
    for i, (tau, rho) in enumerate(batch):
        lo, hi = window(t0[i], tau)
        hits = [m for m in mf_events if _circ_overlap(m.t_lo, m.t_hi, lo, hi, grid.t_obs)]
        m = max(hits, key=lambda m: abs(m.z)) if hits else None
        mf_used |= {id(h) for h in hits}
        rec_mf["found_mf"].append(m is not None)
        rec_mf["z_mf"].append(abs(m.z) if m else np.nan)
        rec_mf["dt_mf"].append(circ(m.t0 - t0[i], grid.t_obs) if m else np.nan)
        s = seeds.get(i)
        rec_mf["seed_dt"].append(circ(s.t0 - t0[i], grid.t_obs) if s else np.nan)
        rec_mf["seed_tau"].append(s.tau if s else np.nan)
        rec_mf["seed_dv_ratio"].append(s.dv / params[i, 1] if s else np.nan)
        rec_mf["seed_rho"].append(s.rho if s else np.nan)
        rec_mf["seed_sigma_t0"].append(s.sigma_t0 if s else np.nan)
        rec_mf["seed_link_exact"].append(bool(s and s.link == LINKS[links[i]]))
        rec_mf["seed_arm_ok"].append(bool(s and s.link in (LINKS[links[i]],
                                                            mfilter.PARTNER[LINKS[links[i]]])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=40)
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--mf", action="store_true",
                    help="also run the matched filter and follow the events up (see docstring)")
    ap.add_argument("--far", type=float, default=FAR,
                    help="false alarms/yr of the excess-power threshold (the matched filter "
                         "stays at 1/yr); != 1 tags the output, e.g. _far10")
    ap.add_argument("--figure", action="store_true", help="redraw from the stored npz only")
    args = ap.parse_args()
    tag = (("_robust" if args.robust else "") + ("_mf" if args.mf else "")
           + (f"_far{args.far:g}" if args.far != FAR else ""))
    if args.figure:
        z = dict(np.load(RES / f"efficiency{tag}.npz"))
        if args.mf:
            figure_mf(z, tag)
        else:
            figure({k: z[k] for k in ("tau", "rho", "found", "dt")}, z["rho_fine"], z["pred"], tag)
        return
    grid = Grid()
    search = Search(grid, robust=args.robust, calibration=load_calibration(RES, args.robust))
    thr = search.threshold(args.far)
    print(grid, search, sep="\n")
    print(f"threshold at FAR = {args.far}/yr: log10 p = {thr / np.log(10):.3f} "
          f"({'calibrated' if search.calibration else 'Bonferroni'})")
    if args.mf:
        mf = mfilter.MatchedFilter(grid, cache=RES / "mf_bank.npz")
        mf_curve = FarCurve.load(RES / "far_calibration_mf.npz")
        thr_mf = mf_curve.threshold(FAR)
        print(mf)
        print(f"matched filter at FAR = {FAR}/yr: log10 p = {thr_mf / np.log(10):.3f}, "
              f"|z| = {np.sqrt(mfilter._chi2_1_isf(thr_mf)):.3f}")
        rec_mf = {k: [] for k in ("found_mf", "z_mf", "dt_mf", "seed_dt", "seed_tau",
                                  "seed_dv_ratio", "seed_rho", "seed_sigma_t0",
                                  "seed_link_exact", "seed_arm_ok")}
        n_false_mf = 0

    rng = np.random.default_rng(args.seed)
    cells = np.array([(t, r) for t in TAUS for r in RHOS for _ in range(args.per_cell)])
    rng.shuffle(cells)
    n_years = int(np.ceil(len(cells) / PER_YEAR))
    rec = {k: [] for k in ("tau", "rho", "found", "log10p", "dt", "rho_ep", "nt")}
    n_false, t1 = 0, time.time()
    for y in range(n_years):
        batch = cells[y * PER_YEAR:(y + 1) * PER_YEAR]
        slots = (np.arange(len(batch)) + 0.5 + rng.uniform(-0.3, 0.3, len(batch))) / PER_YEAR
        t0 = slots * grid.t_obs
        params = with_snr(np.column_stack([t0, np.ones(len(batch)), batch[:, 0]]),
                          batch[:, 1], grid)
        params[:, 1] *= rng.choice([-1.0, 1.0], len(batch))
        links = rng.integers(0, 6, len(batch))
        data = noise_fd(jr.PRNGKey(args.seed + y), grid) + inject(params, links, grid)
        events, _ = search.run(data, log_p_thr=thr)
        used = set()
        hits_ep = {}
        for i, (tau, rho) in enumerate(batch):
            lo, hi = window(t0[i], tau)
            hits = [e for e in events
                    if any(_circ_overlap(c.t_lo, c.t_hi, lo, hi, grid.t_obs) for c in e.core)]
            e = min(hits, key=lambda e: e.logp) if hits else None
            if e is not None:
                hits_ep[i] = e
            used |= {id(h) for h in hits}
            rec["tau"].append(tau)
            rec["rho"].append(rho)
            rec["found"].append(e is not None)
            rec["log10p"].append(e.log10p if e else np.nan)
            d = ((e.t_hat - t0[i] + 0.5 * grid.t_obs) % grid.t_obs - 0.5 * grid.t_obs) if e else np.nan
            rec["dt"].append(d)
            rec["rho_ep"].append(e.rho_ep if e else np.nan)
            rec["nt"].append(e.seed.level.nt if e else 0)
        n_false += sum(id(e) not in used for e in events)
        if args.mf:
            mf_events, mf_scan = mf.run(data, thr_mf)
            idx = sorted(hits_ep)
            found_seeds = mfilter.followup([hits_ep[i] for i in idx], data, mf, scan=mf_scan)
            seeds = dict(zip(idx, found_seeds))
            mf_used = set()
            mf_columns(rec_mf, mf_events, mf_used, seeds, grid, t0, params, links, batch)
            n_false_mf += sum(id(m) not in mf_used for m in mf_events)
            if (y + 1) % 10 == 0:
                print(f"  {y + 1} years, {time.time() - t1:.0f} s", flush=True)
    rec = {k: np.array(v) for k, v in rec.items()}
    print(f"{len(cells)} injections in {n_years} noise years, {time.time() - t1:.0f} s")
    print(f"events matching no injection: {n_false} in {n_years * grid.years:.1f} yr "
          f"-> {n_false / (n_years * grid.years):.2f} per year (target {args.far})")
    if args.mf:
        rec_mf = {k: np.array(v) for k, v in rec_mf.items()}
        print(f"matched-filter events matching no injection: {n_false_mf} "
              f"-> {n_false_mf / (n_years * grid.years):.2f} per year (target {FAR})")
        rec |= rec_mf

    # prediction from noiseless templates
    t0s = design.placements(grid, 16)
    lam, dof = design.search_response(search, TAUS, t0s)
    rho_fine = np.linspace(3, 16, 80)
    pred = np.array([design.detection_probability(rho_fine, lam[i], dof, thr)
                     for i in range(len(TAUS))])

    print(f"\n{'tau [s]':>8} | measured efficiency at rho = " + " ".join(f"{r:5.0f}" for r in RHOS)
          + " | rho_50 meas / pred | median dt [s] | rho_ep/rho | seed nt")
    for i, tau in enumerate(TAUS):
        sel = rec["tau"] == tau
        eff = np.array([rec["found"][sel & (rec["rho"] == r)].mean() for r in RHOS])
        r50_meas = np.interp(0.5, eff, RHOS) if eff.max() >= 0.5 else np.nan
        r50_pred = np.interp(0.5, pred[i], rho_fine)
        det = sel & rec["found"] & (rec["rho"] >= 10)
        nts, cnt = np.unique(rec["nt"][det], return_counts=True)
        print(f"{tau:8.0f} | " + " ".join(f"{x:5.2f}" for x in eff)
              + f" | {r50_meas:5.2f} / {r50_pred:5.2f} | {np.median(rec['dt'][det]):9.0f}    | "
              f"{np.median(rec['rho_ep'][det] / rec['rho'][det]):6.3f}   | "
              + ",".join(f"{n}" for n in nts[np.argsort(-cnt)][:2]))

    extra = {}
    if args.mf:
        extra = dict(log_p_thr_mf=thr_mf, n_false_mf=n_false_mf)
        mf_summary(rec)
    RES.mkdir(exist_ok=True)
    np.savez(RES / f"efficiency{tag}.npz", taus=TAUS, rhos=RHOS, rho_fine=rho_fine, pred=pred,
             n_false=n_false, years=n_years * grid.years, log_p_thr=thr, **rec, **extra)
    print(f"  wrote {RES / f'efficiency{tag}.npz'}")

    if args.mf:
        figure_mf(dict(np.load(RES / f"efficiency{tag}.npz")), tag)
    else:
        figure(rec, rho_fine, pred, tag)


def rho50(found, rho, rng=None):
    """SNR at 50% efficiency, interpolated on the RHOS grid; with rng, one bootstrap draw."""
    if rng is not None:
        i = rng.integers(0, found.size, found.size)
        found, rho = found[i], rho[i]
    eff = np.array([found[rho == r].mean() if np.any(rho == r) else np.nan for r in RHOS])
    ok = np.isfinite(eff)
    eff = np.maximum.accumulate(np.where(ok, eff, 0.0))
    return float(np.interp(0.5, eff, RHOS)) if eff.max() >= 0.5 else np.nan


def rho50_with_error(found, rho, n_boot=400, seed=7):
    rng = np.random.default_rng(seed)
    boot = [rho50(found, rho, rng) for _ in range(n_boot)]
    return rho50(found, rho), float(np.nanstd(boot))


def mf_summary(rec):
    print(f"\n{'tau [s]':>8} | rho_50: excess power, matched filter, ratio | follow-up seeds of "
          "glitches found at rho >= 10: median |dt| (90%), tau ratio, Dv ratio, arm, link, rho_seed/rho")
    for tau in TAUS:
        sel = rec["tau"] == tau
        r_ep, e_ep = rho50_with_error(rec["found"][sel], rec["rho"][sel])
        r_mf, e_mf = rho50_with_error(rec["found_mf"][sel], rec["rho"][sel])
        det = sel & rec["found"] & (rec["rho"] >= 10)
        adt = np.abs(rec["seed_dt"][det])
        print(f"{tau:8.0f} | {r_ep:5.2f}+-{e_ep:.2f} {r_mf:5.2f}+-{e_mf:.2f} {r_ep / r_mf:5.2f} | "
              f"{np.median(adt):8.1f} s ({np.percentile(adt, 90):7.1f}) "
              f"{np.median(rec['seed_tau'][det] / tau):6.3f} "
              f"{np.median(rec['seed_dv_ratio'][det]):6.3f} "
              f"{rec['seed_arm_ok'][det].mean():5.3f} {rec['seed_link_exact'][det].mean():5.3f} "
              f"{np.median(rec['seed_rho'][det] / rec['rho'][det]):6.3f}")


def figure_mf(z, tag="_mf"):
    """(a) rho_50 against tau for the two searches; (b) how well an event is timed, by the
    excess-power centroid alone and after the matched-filter follow-up."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_IN, 2.5))
    r = {k: np.array([rho50_with_error(z[k][z["tau"] == t], z["rho"][z["tau"] == t])
                      for t in TAUS]) for k in ("found", "found_mf")}
    pred50 = np.array([np.interp(0.5, p, z["rho_fine"]) for p in z["pred"]])
    ax1.plot(TAUS, pred50, color=C["blue"], lw=0.9, ls="--", label="excess power, predicted")
    ax1.errorbar(TAUS, r["found"][:, 0], r["found"][:, 1], fmt="o", ms=4, color=C["blue"],
                 lw=0.8, capsize=2, label="excess power, measured")
    ax1.errorbar(TAUS, r["found_mf"][:, 0], r["found_mf"][:, 1], fmt="s", ms=4, color=C["red"],
                 lw=0.8, capsize=2, label="matched filter, measured")
    dz = RES / "design_tiling.npz"
    if dz.exists():
        d = np.load(dz)
        ax1.plot(d["taus"], d["rho50_mf"], color=C["red"], lw=0.9, ls=":",
                 label="matched filter, Rice estimate")
    ax1.set(xscale="log", xlabel=r"decay time $\tau$ [s]", ylabel=r"$\rho_{50}$ at 1 false alarm/yr",
            ylim=(3.5, 10.5), title="(a) SNR for 50% detection")
    ax1.legend(loc="upper left")

    det = z["found"] & (z["rho"] >= 10)
    for key, lab, col, mk, dx in (("dt", r"excess-power centroid, $|\hat t - t_0|$", C["blue"], "o", 0.93),
                                  ("seed_dt", r"matched-filter seed, $|t_0^{\rm seed} - t_0|$",
                                   C["red"], "s", 1.07)):
        q = np.array([np.percentile(np.abs(z[key][det & (z["tau"] == t)]), [50, 90]) for t in TAUS])
        ax2.errorbar(TAUS * dx, q[:, 0], [np.zeros(len(TAUS)), q[:, 1] - q[:, 0]], fmt=mk, ms=4,
                     color=col, lw=0.8, capsize=2, label=lab + " (median, 90%)")
    ax2.axhline(2.6e3, color=C["black"], lw=0.8, ls="--")
    ax2.text(0.8, 3.0e3, "heterodyne tolerance (inference paper)", fontsize=6)
    ax2.set(xscale="log", yscale="log", xlabel=r"decay time $\tau$ [s]", ylim=(1.0, 3e5),
            ylabel="onset error [s]", title=r"(b) onset time, glitches found at $\rho\geq10$")
    ax2.legend(loc="upper left")
    save(fig, FIG / f"efficiency{tag}.pdf")


if __name__ == "__main__":
    main()

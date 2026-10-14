"""Design study: where to cut the data in frequency, and how to tile the year in WDM.

Two questions, each answered from noiseless templates and exact chi^2 laws, no sampling:

band     Fraction of a glitch's rho^2 below a frequency f, for decay times from 0.1 s to
         1e4 s, and the share carried by T. Justifies dt = 60 s (f_Nyq = 8.33 mHz) and the
         A + E statistic.
tiling   SNR needed for 50% detection at a false-alarm rate of 1/yr (Bonferroni over the
         tiles, as the search uses by default), against tau, for every dyadic tiling of the
         year from nt = 512 (17 h bins) to 131072 (4 min), for the five-level LADDER, and for
         all nine levels at once. Justifies LADDER, and SINGLE as the one-tiling fallback.
         An ideal matched filter is drawn for scale: its threshold from Rice's upcrossing
         rate, over a bank of 10 independent tau templates (an estimate, not a search).

    python scripts/design_tiling.py            # both, ~2 min on a CPU
    python scripts/design_tiling.py --banks    # also: do band lower edges m_lo > 1 help?
    python scripts/design_tiling.py --figure   # redraw from results/design_tiling.npz

Writes results/design_tiling.npz and figures/design_tiling.pdf.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from glitchsearch import Grid, Search, LADDER, SINGLE, bank_for, design, wdm  # noqa: E402
from glitchsearch.plotting import plt, save, C, FULL_IN, SERIES  # noqa: E402

RES, FIG = HERE.parent / "results", HERE.parent / "figures"
FAR = 1.0
DYADIC = tuple(2 ** j for j in range(9, 18))
TAUS = np.geomspace(0.1, 5e4, 18)
TAU_BAND = np.array([0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1e3, 3e3, 1e4])
F_CUTS = np.array([1e-3, 2e-3, 3e-3, 5e-3, 1 / 120.0, 1e-2, 2e-2])
N_TAU_TEMPLATES = 10


def band_study():
    f, cum, frac, t_share = design.rho2_below(TAU_BAND, F_CUTS)
    print("\n== band: fraction of rho^2 below f_cut (A + E + T), and T's share ==")
    print(f"{'tau [s]':>9} {'f_knee':>9} | " + " ".join(f"{c * 1e3:6.2f}" for c in F_CUTS)
          + " mHz | T share")
    for tau, fr, ts in zip(TAU_BAND, frac, t_share):
        print(f"{tau:9.1f} {1e3 / (2 * np.pi * tau):7.3f}mHz | "
              + " ".join(f"{x:6.3f}" for x in fr) + f"     | {ts:.4f}")
    return dict(band_f=f[::200], band_cum=cum[:, ::200], band_tau=TAU_BAND, band_fcut=F_CUTS,
                band_frac=frac, band_tshare=t_share)


def mf_estimate(grid, taus):
    """Matched-filter 50% threshold: Rice rate 2 f_rms T exp(-z^2/2) per template, times
    N_TAU_TEMPLATES, set equal to FAR * years."""
    z = []
    for tau in taus:
        h = design.simulate.glitch_fd([0.0, 1.0, float(tau)], grid)
        Y2 = np.asarray(jnp.sum(jnp.abs(wdm.whiten(h, grid)) ** 2, axis=0))
        f_rms = np.sqrt(np.sum(grid.freq ** 2 * Y2) / np.sum(Y2))
        rate = 2.0 * f_rms * grid.t_obs * N_TAU_TEMPLATES
        z.append(np.sqrt(2.0 * np.log(rate / (FAR * grid.years))))
    return np.array(z)


def tiling_study(grid, t0s):
    print("\n== tiling: rho for 50% detection at FAR = 1/yr (Bonferroni), median over "
          f"{len(t0s)} placements ==")
    resp, nshape = {}, {}
    for nt in DYADIC:
        lev = grid.level(nt)
        bank = bank_for(lev)
        resp[nt] = design.level_response(grid, lev, bank, TAUS, t0s)
        nshape[nt] = len(bank)
        print(f"  computed nt = {nt}", flush=True)

    def rho50(nts):
        lam = np.concatenate([resp[nt][0] for nt in nts], axis=2)
        dof = np.concatenate([resp[nt][1] for nt in nts])
        thr = np.log(FAR * grid.years / sum(nt * nshape[nt] for nt in nts))
        return np.array([design.rho_for(0.5, lam[i], dof, thr) for i in range(len(TAUS))])

    configs = {f"nt={nt}": (nt,) for nt in DYADIC}
    configs["LADDER"] = LADDER
    configs["all 9 dyadic"] = DYADIC
    table = {name: rho50(nts) for name, nts in configs.items()}
    best = np.min([table[f"nt={nt}"] for nt in DYADIC], axis=0)
    print(f"{'':>14}" + "".join(f"{t:6.3g}" for t in TAUS) + "   worst loss vs best single")
    for name, r in table.items():
        sel = (TAUS >= 1) & (TAUS <= 1e4)
        print(f"{name:>14}" + "".join(f"{x:6.1f}" for x in r)
              + f"   {100 * (r[sel] / best[sel] - 1).max():6.1f}% (tau 1..1e4 s)")
    z_mf = mf_estimate(grid, TAUS)
    print(f"{'matched filt.':>14}" + "".join(f"{x:6.1f}" for x in z_mf)
          + f"   (Rice estimate, {N_TAU_TEMPLATES} tau templates)")
    return dict(taus=TAUS, t0s=t0s, configs=np.array(list(configs)),
                rho50=np.array([table[k] for k in configs]), rho50_mf=z_mf)


def bank_study(grid, t0s):
    """Band designs on the ladder: low-pass only (the search), or with lower edges too."""
    print("\n== banks: rho_50 on the LADDER for three band designs ==")
    designs = {"[1, m_hi] (search)": (0.0,), "+ [m_hi/4, m_hi]": (0.0, 0.25),
               "+ [m_hi/8, m_hi], [m_hi/3, m_hi]": (0.0, 1 / 8, 1 / 3)}
    out = {}
    for name, fracs in designs.items():
        lam_all, dof_all, n_trials = [], [], 0
        for nt in LADDER:
            lev = grid.level(nt)
            his = sorted({s.m_hi for s in bank_for(lev)})
            bands = sorted({(max(1, int(np.ceil(h * fr))), h) for h in his for fr in fracs})
            lam = np.array([[design.captures(design.unit_power(grid, lev, tau, t0 - 2 * tau),
                                             bands, (1, 2)) for t0 in t0s] for tau in TAUS])
            dof = np.array([2 * k * (hi - lo + 1) for k in (1, 2) for lo, hi in bands], float)
            lam_all.append(lam)
            dof_all.append(dof)
            n_trials += nt * dof.size
        lam, dof = np.concatenate(lam_all, axis=2), np.concatenate(dof_all)
        thr = np.log(FAR * grid.years / n_trials)
        out[name] = np.array([design.rho_for(0.5, lam[i], dof, thr) for i in range(len(TAUS))])
        print(f"{name:>34}" + "".join(f"{x:6.1f}" for x in out[name]), flush=True)
    return dict(bank_designs=np.array(list(out)), bank_rho50=np.array(list(out.values())))


def figure(res):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_IN, 2.6))
    show = [0.1, 10.0, 100.0, 300.0, 1e3, 1e4]
    for i, tau in enumerate(show):
        j = int(np.argmin(np.abs(res["band_tau"] - tau)))
        ax1.semilogx(res["band_f"], res["band_cum"][j], color=SERIES[i],
                     ls=["-", "--", "-.", ":", "-", "--"][i], label=f"$\\tau$ = {tau:g} s")
    for f_line, lab in ((3e-3, "3 mHz (inference paper)"), (1 / 120.0, "8.33 mHz (this grid)")):
        ax1.axvline(f_line, color=C["grey"], lw=0.8, ls=":")
        ax1.text(f_line * 1.08, 0.04, lab, rotation=90, fontsize=6, color=C["grey"],
                 va="bottom")
    ax1.set(xlabel="frequency $f$ [Hz]", ylabel=r"fraction of $\rho^2$ below $f$",
            xlim=(1e-5, 1e-1), ylim=(0, 1.02), title="(a) where a glitch's SNR lives")
    ax1.legend(loc="upper left")

    cmap = plt.get_cmap("Blues")
    names = list(res["configs"])
    for i, nt in enumerate(DYADIC):
        r = res["rho50"][names.index(f"nt={nt}")]
        lw, ls = (1.3, "--") if (nt,) == SINGLE else (0.8, "-")
        ax2.loglog(res["taus"], r, color=cmap(0.3 + 0.7 * i / (len(DYADIC) - 1)), lw=lw, ls=ls,
                   label=(f"single, $\\Delta t$ = {Grid().t_obs / nt / 60:.0f} min"
                          if (nt,) == SINGLE else None))
    ax2.loglog(res["taus"], res["rho50"][names.index("LADDER")], color=C["black"], lw=2,
               marker="o", ms=3, label="five-level ladder (search)")
    ax2.loglog(res["taus"], res["rho50_mf"], color=C["red"], lw=1.2, ls=":",
               label="ideal matched filter (estimate)")
    ax2.text(0.13, 30, "single tilings,\n$\\Delta t$ = 17 h (light)\nto 4 min (dark)",
             fontsize=6, color=C["grey"])
    ax2.set(xlabel=r"decay time $\tau$ [s]", ylabel=r"$\rho$ for 50% detection at 1/yr",
            ylim=(4, 200), title="(b) one tiling or a ladder")
    ax2.legend(loc="upper right")
    save(fig, FIG / "design_tiling.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banks", action="store_true", help="also compare band designs")
    ap.add_argument("--figure", action="store_true", help="redraw from the stored npz only")
    args = ap.parse_args()
    if args.figure:
        figure(dict(np.load(RES / "design_tiling.npz")))
        return
    grid = Grid()
    t0s = design.placements(grid, 16)
    print(grid)
    print(Search(grid))
    res = band_study()
    res |= tiling_study(grid, t0s)
    if args.banks:
        res |= bank_study(grid, t0s)
    RES.mkdir(exist_ok=True)
    np.savez(RES / "design_tiling.npz", **res)
    print(f"  wrote {RES / 'design_tiling.npz'}")
    figure(res)


if __name__ == "__main__":
    main()

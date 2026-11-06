#!/usr/bin/env python3
"""
Residue-to-residue distance from smFRET data ONLY (Alexa555 donor / Alexa647 acceptor).
No structure needed.

What it does
------------
1. Converts each FRET efficiency E to the apparent dye-dye distance:
        R_DA = R0 * (1/E - 1)^(1/6)
2. Corrects for the dye linkers (maleimide-cysteine attachment). Each dye is
   modelled as freely sampling a sphere around its attachment point (the
   C-beta/C-alpha of the labeled residue) with radius = effective linker length.
   For every separation d between the two attachment points the mean efficiency
        <E>(d) = < 1 / (1 + (r/R0)^6) >
   is computed (averaging over all dye-position pairs, as happens in a real
   experiment with fast dye motion). This calibration curve is inverted to get
   the distance d between the two labeled residues that reproduces your E.
3. Reports uncertainty from E_err and R0 uncertainty.

Input
-----
Either a CSV with columns   state,E[,E_err]      e.g.
        open,0.35,0.03
        closed,0.78,0.02
or values on the command line:
        python smfret_residue_distance.py --E 0.35 0.78 --E_err 0.03 0.02 --names open closed

Requirements:  pip install numpy pandas matplotlib

Parameters (EDIT if you have better numbers for your labeling chemistry)
-----------------------------------------------------------------------
R0 (Alexa555/Alexa647, kappa^2 = 2/3) ~ 51 A (literature range ~ 50-52 A, buffer dependent).
Effective linker lengths (maleimide + C5/C6 linker, from FRET-community AV
parameter sets, approximate): donor ~ 20 A, acceptor ~ 22 A.
Please check these against the values for your own dyes (e.g. Hellenkamp et al.
2018 Nat. Methods; FPS software) and override with --L_D / --L_A / --R0.

IMPORTANT: E must be fully corrected (leakage, direct excitation, gamma).
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------- default dye properties
R0_DEFAULT = 51.0        # A, Alexa555/Alexa647
R0_ERR_DEFAULT = 2.0     # A
L_DONOR_DEFAULT = 20.0   # A, Alexa555 effective linker length
L_ACCEPTOR_DEFAULT = 22.0  # A, Alexa647 effective linker length



def read_table(path):
    """Read a CSV in either format: comma/dot (international) or semicolon/decimal-comma (European Excel)."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        first = fh.readline()
    if first.count(";") > first.count(","):
        return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    return pd.read_csv(path, encoding="utf-8-sig")

# ------------------------------------------------------------------ FRET maths
def E_to_R(E, R0):
    E = np.asarray(E, dtype=float)
    return R0 * (1.0 / E - 1.0) ** (1.0 / 6.0)


def R_to_E(R, R0):
    return 1.0 / (1.0 + (np.asarray(R, dtype=float) / R0) ** 6)


def sample_sphere(radius, n, rng):
    """Uniform random points inside a sphere centered on the origin."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return v * (radius * rng.random(n) ** (1 / 3))[:, None]


# --------------------------------------------------------------- calibration curve
def make_difference_vectors(L_D, L_A, n_pairs=400_000, seed=0):
    """Vectors (acceptor position - donor position) when both attachment points coincide."""
    rng = np.random.default_rng(seed)
    return sample_sphere(L_A, n_pairs, rng) - sample_sphere(L_D, n_pairs, rng)


def calibration_curve(R0, diff_vecs, d_max=160.0, step=0.5):
    """<E> as a function of attachment-point separation d (A)."""
    d_grid = np.arange(0.0, d_max + step, step)
    E_grid = np.empty_like(d_grid)
    for i, d in enumerate(d_grid):
        v = diff_vecs.copy()
        v[:, 2] += d
        r = np.linalg.norm(v, axis=1)
        E_grid[i] = R_to_E(r, R0).mean()
    return d_grid, E_grid


def invert(E, d_grid, E_grid):
    """Find d such that <E>(d) = E. Returns (d, flag) with flag for out-of-range E."""
    flag = ""
    if E > E_grid[0]:
        flag = "E too high for model (clamped to d=0)"
    elif E < E_grid[-1]:
        flag = "E too low for model (clamped to d_max)"
    return float(np.interp(E, E_grid[::-1], d_grid[::-1])), flag


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="CSV with columns state,E[,E_err]")
    ap.add_argument("--E", nargs="+", type=float, help="efficiencies (instead of CSV)")
    ap.add_argument("--E_err", nargs="+", type=float, help="errors on E")
    ap.add_argument("--names", nargs="+", help="state names")
    ap.add_argument("--R0", type=float, default=R0_DEFAULT)
    ap.add_argument("--R0_err", type=float, default=R0_ERR_DEFAULT)
    ap.add_argument("--L_D", type=float, default=L_DONOR_DEFAULT, help="donor linker length (A)")
    ap.add_argument("--L_A", type=float, default=L_ACCEPTOR_DEFAULT, help="acceptor linker length (A)")
    ap.add_argument("--out", default="smfret_residue_distances")
    args = ap.parse_args()

    # ---- read data
    if args.csv:
        df = read_table(args.csv)
    elif args.E:
        df = pd.DataFrame({"E": args.E})
        df["state"] = args.names if args.names else [f"state{i+1}" for i in range(len(args.E))]
        df["E_err"] = args.E_err if args.E_err else 0.0
    else:
        ap.error("Provide a CSV file or --E values")
    if "E_err" not in df:
        df["E_err"] = 0.0

    # ---- calibration curves for R0 -/nominal/+ (to propagate R0 uncertainty)
    diff = make_difference_vectors(args.L_D, args.L_A)
    curves = {k: calibration_curve(args.R0 + k * args.R0_err, diff) for k in (-1, 0, 1)}
    d0, E0 = curves[0]

    has_pop = "fraction" in df.columns
    rows = []
    for _, r in df.iterrows():
        E, dE = float(r.E), float(r.E_err)
        if not 0 < E < 1:
            raise ValueError(f"E must be between 0 and 1 (got {E} for {r.state})")
        d_best, flag = invert(E, d0, E0)
        if E < 0.15 or E > 0.85:
            flag = (flag + "; " if flag else "") + "E outside 0.15-0.85: distance only qualitative (Lee 2005)"
        # range over E +/- dE and R0 +/- R0_err
        cands = []
        for k in (-1, 0, 1):
            dg, Eg = curves[k]
            for e in (max(E - dE, 1e-3), E, min(E + dE, 0.999)):
                cands.append(invert(e, dg, Eg)[0])
        row = dict(state=r.state, E=E, E_err=dE)
        if has_pop:
            row["population_pct"] = 100.0 * float(r.fraction)
            if "fraction_err" in df.columns:
                row["population_err_pct"] = 100.0 * float(r.fraction_err)
        row.update(dye_dye_R_A=float(E_to_R(E, args.R0)),
                   residue_residue_distance_A=d_best,
                   distance_min_A=min(cands), distance_max_A=max(cands), note=flag)
        rows.append(row)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(f"\nR0 = {args.R0} +/- {args.R0_err} A | linker donor = {args.L_D} A, acceptor = {args.L_A} A\n")
    show = res.copy()
    for c in ("E", "E_err"):
        show[c] = show[c].round(3)
    print(show.round(1).assign(E=show.E, E_err=show.E_err).to_string(index=False))
    res.to_csv(args.out + ".csv", index=False)

    if has_pop:
        print("\nPopulation of each distance state (share of bursts in the fit):")
        for _, r in res.sort_values("residue_residue_distance_A").iterrows():
            err = f" +/- {r.population_err_pct:.1f}" if "population_err_pct" in res else ""
            print(f"   {r.residue_residue_distance_A:5.1f} A  "
                  f"({r.distance_min_A:.0f}-{r.distance_max_A:.0f})  ->  {r.population_pct:5.1f}{err} %   [{r.state}]")

    # ---- plot: calibration curve + measured states (+ population panel)
    ncol = 2 if has_pop else 1
    fig, axs = plt.subplots(1, ncol, figsize=(6 * ncol, 4.5))
    ax = axs[0] if has_pop else axs
    ax.plot(d0, E0, "k-", label="Model <E>(d), linker-averaged")
    ax.plot(d0, R_to_E(d0, args.R0), "k--", alpha=0.5, label="Point dyes (no linker)")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (_, r) in enumerate(res.iterrows()):
        lab = r.state + (f" ({r.population_pct:.0f} %)" if has_pop else "")
        ax.errorbar(r.residue_residue_distance_A, r.E,
                    xerr=[[r.residue_residue_distance_A - r.distance_min_A],
                          [r.distance_max_A - r.residue_residue_distance_A]],
                    yerr=r.E_err, fmt="o", capsize=3, color=colors[i % len(colors)], label=lab)
    ax.set_xlabel("Distance between labeled residues (Å)")
    ax.set_ylabel("FRET efficiency")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8)
    if has_pop:
        ax2 = axs[1]
        for i, (_, r) in enumerate(res.iterrows()):
            c = colors[i % len(colors)]
            yerr = r.population_err_pct if "population_err_pct" in res else 0
            ax2.bar(r.residue_residue_distance_A, r.population_pct, width=3.0, color=c, alpha=0.8,
                    yerr=yerr, capsize=3)
            ax2.errorbar(r.residue_residue_distance_A, r.population_pct * 0.5,
                         xerr=[[r.residue_residue_distance_A - r.distance_min_A],
                               [r.distance_max_A - r.residue_residue_distance_A]],
                         fmt="none", ecolor="k", capsize=3, lw=1)
            ax2.text(r.residue_residue_distance_A, r.population_pct + (yerr if yerr == yerr else 0) + 1.5,
                     f"{r.population_pct:.0f} %", ha="center", fontsize=9)
        ax2.set_xlabel("Distance between labeled residues (Å)")
        ax2.set_ylabel("Population (% of bursts)")
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, min(115, max(res.population_pct) * 1.25 + 8))
    plt.tight_layout()
    plt.savefig(args.out + ".png", dpi=200)
    print(f"\nSaved {args.out}.csv and {args.out}.png")


if __name__ == "__main__":
    main()

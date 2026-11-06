#!/usr/bin/env python3
"""
smFRET efficiency -> inter-dye distance, compared with distances from static structures.

For every row (conformational state) in the input CSV the script:
  1. converts the experimental FRET efficiency E to a distance R (with error propagation)
  2. measures CA-CA and CB-CB distances between your labeled residues in the PDB
  3. (optional) estimates dye-dye distances with a simplified Accessible Volume (AV) model
     and predicts the mean FRET efficiency <E> for that structure
  4. writes a results table (CSV) and a comparison plot (PNG)

INPUT CSV columns (header required):
  state,pdb_file,chain1,res1,chain2,res2,E,E_err
Example:
  open,1abc_open.pdb,A,52,A,210,0.35,0.03
  closed,1abc_closed.pdb,A,52,A,210,0.78,0.02

NOTE: E must be a *fully corrected* efficiency (donor leakage, direct acceptor
excitation, and gamma factor) or the distances will be wrong.

Requirements:  pip install numpy pandas scipy matplotlib biopython

Usage:
  python smfret_vs_structure.py data.csv --R0 51 --av --linker 20 --dye-radius 3.5
(R0 in Angstrom. Typical: Alexa488/594 ~60, Cy3/Cy5 ~54-56, Alexa555/647 ~51.
 Use the R0 measured/reported for YOUR dye pair and buffer.)
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser
from Bio import BiopythonWarning

warnings.simplefilter("ignore", BiopythonWarning)



def read_table(path):
    """Read CSV in international (comma/dot) or European Excel (semicolon/decimal-comma) format."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        first = fh.readline()
    if first.count(";") > first.count(","):
        return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    return pd.read_csv(path, encoding="utf-8-sig")

# ----------------------------------------------------------------- FRET maths
def E_to_R(E, R0):
    """Distance (A) from efficiency: R = R0 * (1/E - 1)^(1/6)."""
    E = np.asarray(E, dtype=float)
    return R0 * (1.0 / E - 1.0) ** (1.0 / 6.0)


def R_to_E(R, R0):
    return 1.0 / (1.0 + (np.asarray(R, dtype=float) / R0) ** 6)


def R_error(E, dE, R0):
    """Propagated error on R from error on E (ignores R0 uncertainty)."""
    E = np.asarray(E, dtype=float)
    dRdE = -R0 / 6.0 * (1.0 / E - 1.0) ** (-5.0 / 6.0) / E ** 2
    return np.abs(dRdE) * dE


# ------------------------------------------------------------ structure parts
_parser = PDBParser(QUIET=True)


def load_structure(path):
    return _parser.get_structure(path, path)[0]  # first model


def get_residue(model, chain, resnum):
    try:
        return model[chain][int(resnum)]
    except KeyError:
        raise KeyError(f"Residue {chain}:{resnum} not found in structure")


def atom_coord(res, name):
    if name == "CB" and "CB" not in res:  # glycine
        name = "CA"
    return res[name].coord.astype(float)


def sample_sphere(center, radius, n, rng):
    """Uniform random points inside a sphere."""
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    r = radius * rng.random(n) ** (1 / 3)
    return center + v * r[:, None]


def accessible_volume(model, res, linker, dye_radius, n_samples=20000, seed=0):
    """
    Simplified AV: random points within `linker` (+ dye radius) of the CB atom that
    (a) do not clash with protein heavy atoms (distance > dye_radius) and
    (b) are reachable by a straight, unobstructed line from the attachment atom.
    Atoms of the labeled residue itself are ignored. This is a simplification of
    the Kalinin et al. 2012 AV approach (use FPS / FRETraj / PyMOL for production work).
    """
    rng = np.random.default_rng(seed)
    atoms = [a for a in model.get_atoms()
             if a.element != "H" and a.get_parent() is not res
             and a.get_parent().id[0] == " "]   # skip water / hetero groups
    tree = cKDTree(np.array([a.coord for a in atoms]))
    origin = atom_coord(res, "CB")
    pts = sample_sphere(origin, linker, n_samples, rng)

    # (a) clash with protein
    d, _ = tree.query(pts)
    pts = pts[d > dye_radius]

    # (b) straight-line accessibility
    keep = np.ones(len(pts), bool)
    for t in np.linspace(0.15, 1.0, 12):  # skip the very start (inside own residue)
        probe = origin + (pts - origin) * t
        dd, _ = tree.query(probe)
        keep &= dd > dye_radius * 0.8
    pts = pts[keep]
    return pts


def av_distances(pts1, pts2, R0, max_pts=2500, seed=1):
    """Mean distance <R_DA>, mean-FRET-derived distance <R_E>, and <E> between two clouds."""
    rng = np.random.default_rng(seed)
    if len(pts1) == 0 or len(pts2) == 0:
        return np.nan, np.nan, np.nan, np.nan
    a = pts1[rng.choice(len(pts1), min(max_pts, len(pts1)), replace=False)]
    b = pts2[rng.choice(len(pts2), min(max_pts, len(pts2)), replace=False)]
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).ravel()
    R_mean = d.mean()
    E_mean = R_to_E(d, R0).mean()
    R_E = E_to_R(E_mean, R0)
    return R_mean, R_E, E_mean, d.std()


def build_calibration(R0, linker, n_pts=1200, d_max=150.0, step=1.0, seed=2):
    """
    Free-space calibration: <E> as a function of the separation d between the two
    ATTACHMENT points, when each dye can sample a sphere of radius `linker` around
    its attachment point (protein obstruction ignored).
    Returns (d_grid, E_mean_grid).
    """
    rng = np.random.default_rng(seed)
    a = sample_sphere(np.zeros(3), linker, n_pts, rng)
    b = sample_sphere(np.zeros(3), linker, n_pts, rng)
    d_grid = np.arange(0.0, d_max + step, step)
    E_grid = np.empty_like(d_grid)
    for i, d in enumerate(d_grid):
        shift = np.array([0.0, 0.0, d])
        diff = np.linalg.norm(a[:, None, :] - (b + shift)[None, :, :], axis=2)
        E_grid[i] = R_to_E(diff, R0).mean()
    return d_grid, E_grid


def E_to_attach_distance(E, d_grid, E_grid):
    """Invert the calibration curve (E decreases monotonically with d)."""
    return float(np.interp(E, E_grid[::-1], d_grid[::-1]))


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--R0", type=float, default=51.0, help="Forster radius in Angstrom")
    ap.add_argument("--av", action="store_true", help="also run simplified accessible-volume model")
    ap.add_argument("--linker", type=float, default=20.0,
                    help="effective dye linker length in A (incl. dye size), default 20")
    ap.add_argument("--dye-radius", type=float, default=3.5,
                    help="dye clash radius in A, default 3.5")
    ap.add_argument("--calibrate", action="store_true",
                    help="estimate the residue-residue (attachment point) distance from E, "
                         "correcting for the dye linkers (free-space model)")
    ap.add_argument("--pdb", help="PDB file (fills pdb_file column if missing)")
    ap.add_argument("--site1", help="labeled residue 1 as CHAIN:RESNUM, e.g. A:150")
    ap.add_argument("--site2", help="labeled residue 2 as CHAIN:RESNUM, e.g. B:150")
    ap.add_argument("--out", default="smfret_results")
    args = ap.parse_args()

    df = read_table(args.csv)
    if args.pdb:
        df["pdb_file"] = args.pdb
    for tag, site in (("1", args.site1), ("2", args.site2)):
        if site:
            ch, rn = site.split(":")
            df["chain" + tag], df["res" + tag] = ch, int(rn)
    rows = []
    if args.calibrate:
        d_grid, E_grid = build_calibration(args.R0, args.linker)
    for _, r in df.iterrows():
        model = load_structure(r.pdb_file)
        r1 = get_residue(model, r.chain1, r.res1)
        r2 = get_residue(model, r.chain2, r.res2)

        R_exp = float(E_to_R(r.E, args.R0))
        dR = float(R_error(r.E, r.get("E_err", 0.0), args.R0)) if "E_err" in df else np.nan

        out = dict(
            state=r.state, E_exp=r.E, E_err=r.get("E_err", np.nan),
            R_exp_A=R_exp, R_exp_err_A=dR,
            CA_CA_A=np.linalg.norm(atom_coord(r1, "CA") - atom_coord(r2, "CA")),
            CB_CB_A=np.linalg.norm(atom_coord(r1, "CB") - atom_coord(r2, "CB")),
        )
        out["E_from_CB_CB"] = float(R_to_E(out["CB_CB_A"], args.R0))

        if args.av:
            p1 = accessible_volume(model, r1, args.linker, args.dye_radius)
            p2 = accessible_volume(model, r2, args.linker, args.dye_radius)
            R_mean, R_E, E_av, sd = av_distances(p1, p2, args.R0)
            out.update(AV_n1=len(p1), AV_n2=len(p2), AV_R_mean_A=R_mean,
                       AV_R_E_A=R_E, AV_E_pred=E_av, AV_R_sd_A=sd,
                       delta_R_exp_minus_AV_A=R_exp - R_E)
        out["delta_R_exp_minus_CBCB_A"] = R_exp - out["CB_CB_A"]

        if args.calibrate:
            d_est = E_to_attach_distance(r.E, d_grid, E_grid)
            out["Est_residue_dist_A"] = d_est
            if "E_err" in df:
                d_hi = E_to_attach_distance(max(r.E - r.E_err, 1e-3), d_grid, E_grid)
                d_lo = E_to_attach_distance(min(r.E + r.E_err, 0.999), d_grid, E_grid)
                out["Est_residue_dist_err_A"] = 0.5 * (d_hi - d_lo)
            out["delta_Est_minus_CBCB_A"] = d_est - out["CB_CB_A"]
        rows.append(out)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(2).to_string(index=False))
    res.to_csv(args.out + ".csv", index=False)

    # ---- plot
    x = np.arange(len(res))
    series = [("smFRET dye-dye (R from E)", res.R_exp_A, res.R_exp_err_A)]
    if args.calibrate:
        series.append(("smFRET residue-residue (linker-corrected)", res.Est_residue_dist_A,
                       res.get("Est_residue_dist_err_A")))
    series.append(("Structure CB-CB", res.CB_CB_A, None))
    if args.av:
        series.append(("Structure AV <R_E> (dye-dye)", res.AV_R_E_A, res.AV_R_sd_A))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(2.2 * len(res) + 3, 4.5))
    for i, (lab, vals, err) in enumerate(series):
        ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, yerr=err, label=lab, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(res.state)
    ax.set_ylabel("Distance (Å)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out + ".png", dpi=200)
    print(f"\nSaved {args.out}.csv and {args.out}.png")


if __name__ == "__main__":
    main()

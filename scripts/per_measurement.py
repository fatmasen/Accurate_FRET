#!/usr/bin/env python3
"""
Analyse every measurement of a state ONE BY ONE (each with its own APBS reference), decide which
measurements are compatible, and MERGE the compatible ones into one corrected E histogram / state fit.

Folder layout (one folder per state, every measurement = one *_apbs.csv + one *_dcbs.csv):
  data/TF_E96A160/Monomer/ 240502_..._apbs.csv  240502_..._dcbs.csv  250312_..._apbs.csv  ...
Pairs are matched by file name after removing "apbs"/"dcbs" and punctuation
("..._monomer_dcbs_.csv" matches "..._monomer_apbs.csv").

For EVERY measurement
  alpha (leakage)         from its donor-only bursts          (S_app > 0.9)
  delta (direct excitation) from its acceptor-only bursts     (S_app < 0.2)
  gamma                   from the E-S relation of its FRET bursts (Lee et al. 2005, Eq. 20); if that
                          regression is unreliable (narrow E range / large error) the common gamma of the
                          reliable measurements is used instead (gamma depends on detection + dye environment,
                          not on laser power), and the table says so
  beta                    beta = sum(AA) / sum(gamma*DD + DA')  of the FRET bursts (S_gamma = 1/(1+beta))
  -> its DCBS bursts are corrected with ITS OWN factors, E and S are computed.
  delta and beta change with the donor/acceptor laser power ratio; they are NOT a reason to exclude a
  measurement (each is corrected with its own values) but are flagged in the table ("settings").

Compatibility (what is merged)
  A measurement is excluded when
    - its reference has too few donor-only / acceptor-only bursts, or it has < --min_bursts selected bursts, or
    - its corrected E distribution differs from the other measurements: Wasserstein distance (in E units,
      0.07 = 7 % of the E scale) to the pool of the remaining measurements > --w1_max.
      The worst measurement is removed, the pool is re-computed, and so on until all are consistent.
  Manual override: --exclude substr [substr ...]   --include substr [substr ...] (force-keep)

Merging: corrected bursts of the compatible measurements are pooled; the number of states (1-3) is chosen
by BIC as in correct_and_fit.py (or fixed with --n). Errors: bootstrap over MEASUREMENTS, re-drawing the
correction factors of each measurement from its own bootstrap, so the error contains the uncertainty of all
correction factors.

Usage
  python scripts/per_measurement.py --folder TF_E96A160/Monomer
  python scripts/per_measurement.py --folder TF_E96A160/Monomer --exclude nomg2 highS --max_n 2
Output (results/<folder name>/): measurement_summary.csv, apbs_uncorrected.png, uncorrected_vs_corrected.png, factors_per_measurement.png,
  per_measurement_histograms.png, merged_FRET.png, merged_corrected_bursts.csv, data.csv, and
  residue_distances.csv/.png (merged E -> distance, accurate-FRET distance plot)
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 2D histograms in the standard "viridis" colour code (dark purple = empty, yellow = most frequent)
def h2(ax, x, y, **kw):
    kw.pop("cmap", None); kw.pop("cmin", None)
    ax.set_facecolor(plt.get_cmap("viridis")(0.0))
    return ax.hist2d(x, y, cmap="viridis", **kw)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "scripts" else HERE.parents[1]
_spec = importlib.util.spec_from_file_location("cf", HERE / "correct_and_fit.py")
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)


# ------------------------------------------------------------------ files
def key_of(path):
    import re
    s = path.stem.lower().replace("apbs", "").replace("dcbs", "")
    return re.sub(r"[^a-z0-9]", "", s)


def find_pairs(folder):
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    refs = {key_of(p): p for p in files if "apbs" in p.name.lower()}
    tgts = {key_of(p): p for p in files if "dcbs" in p.name.lower()}
    pairs = [(k, tgts[k], refs[k]) for k in sorted(tgts) if k in refs]
    lonely = [p.name for k, p in tgts.items() if k not in refs] + [p.name for k, p in refs.items() if k not in tgts]
    return pairs, lonely


def read_counts(path, bg):
    df = cf.read_table(path)
    miss = [c for c in ("n_DD", "n_DA", "n_AA", "time_length") if c not in df.columns]
    if miss:
        raise ValueError(f"{path.name} lacks columns {miss}")
    DD, DA, AA = cf.counts(df, bg)
    ok = np.isfinite(DD) & np.isfinite(DA) & np.isfinite(AA)
    return df[ok].reset_index(drop=True), DD[ok], DA[ok], AA[ok]


# ------------------------------------------------------------------ factors
def beta_given_gamma(DD, DA, AA, alpha, delta, gamma, args):
    S = cf.S_app(DD, DA, AA)
    f = (S > args.S_fret_lo) & (S < args.S_fret_hi)
    dac = DA[f] - alpha * DD[f] - delta * AA[f]
    den = gamma * DD[f] + dac
    ok = den > 0
    return AA[f][ok].sum() / den[ok].sum() if ok.sum() > 20 else np.nan


def quick_factors(DD, DA, AA, args):
    """alpha, delta, regression gamma, lever arm (E_PR span of the FRET bursts)."""
    S = cf.S_app(DD, DA, AA)
    d, a = S > args.S_donor_only, S < args.S_acceptor_only
    alpha = DA[d].sum() / DD[d].sum()
    delta = DA[a].sum() / AA[a].sum()
    _, _, g, _ = cf.derive_factors(DD, DA, AA, args)
    span = cf.fret_lever_arm(DD, DA, AA, alpha, delta, args)
    return alpha, delta, g, span


def analyse_reference(DD, DA, AA, args, n_boot, rng):
    n = len(DD)
    S = cf.S_app(DD, DA, AA)
    n_do, n_ao = int((S > args.S_donor_only).sum()), int((S < args.S_acceptor_only).sum())
    out = dict(n_ref=n, n_donor_only=n_do, n_acceptor_only=n_ao, ok=(n_do >= 20 and n_ao >= 20))
    if not out["ok"]:
        return out
    al, de, g, span = quick_factors(DD, DA, AA, args)
    boots, idxs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            a_, d_, g_, _s = quick_factors(DD[i], DA[i], AA[i], args)
        except Exception:
            continue
        if np.isfinite([a_, d_, g_]).all():
            boots.append((a_, d_, g_))
            idxs.append(i)
    out.update(alpha=al, delta=de, gamma_reg=g, span=span, boot=np.array(boots), idx=idxs)
    out["gamma_reg_sd"] = out["boot"][:, 2].std() if len(boots) > 10 else np.nan
    out["gamma_reliable"] = bool(span >= 0.4 and np.isfinite(out["gamma_reg_sd"]) and out["gamma_reg_sd"] <= 0.15
                                 and 0.3 < g < 4 and not (g == 1.0))
    return out


# ------------------------------------------------------------------ small helpers
def w1(x, y, nq=199):
    q = np.linspace(0.005, 0.995, nq)
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def state_string(x, args):
    try:
        w, mu, sd, _ = cf.select_model(x, args.max_n, args.min_frac, args.min_sep, args.delta_bic, verbose=False)
    except Exception:
        return ""
    return ", ".join(f"{m:.2f} ({100 * p:.0f}%)" for m, p in zip(mu, w))


def matches(name, subs):
    return any(s.lower() in name.lower() for s in (subs or []))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True, help="state folder (inside data/ or a path)")
    ap.add_argument("--out", help="output folder (default: results/<folder name>)")
    ap.add_argument("--exclude", nargs="+", help="name parts of measurements to exclude")
    ap.add_argument("--include", nargs="+", help="name parts of measurements to keep no matter what")
    ap.add_argument("--w1_max", type=float, default=0.07, help="max Wasserstein distance (E units) to the pool")
    ap.add_argument("--min_bursts", type=int, default=300, help="minimum selected bursts per measurement")
    ap.add_argument("--n", default="auto", help="'auto' or fixed number of states of the merged fit")
    ap.add_argument("--max_n", type=int, default=3)
    ap.add_argument("--min_frac", type=float, default=0.10)
    ap.add_argument("--min_sep", type=float, default=0.15)
    ap.add_argument("--delta_bic", type=float, default=10.0)
    ap.add_argument("--names", nargs="+", help="state names low E -> high E (if the count matches)")
    ap.add_argument("--gamma", type=float, help="use this gamma for ALL measurements (e.g. a measured value)")
    ap.add_argument("--gamma_err", type=float, default=0.0)
    ap.add_argument("--gamma_tol", type=float, default=0.15,
                    help="own regression gamma is used only if within this fraction of the common gamma (gamma does not depend on laser power)")
    ap.add_argument("--S_donor_only", type=float, default=0.9)
    ap.add_argument("--S_acceptor_only", type=float, default=0.2)
    ap.add_argument("--S_fret_lo", type=float, default=0.2)
    ap.add_argument("--S_fret_hi", type=float, default=0.8)
    ap.add_argument("--S_min", type=float, default=0.3)
    ap.add_argument("--S_max", type=float, default=0.7)
    ap.add_argument("--bg_DD", type=float, default=0.0)
    ap.add_argument("--bg_DA", type=float, default=0.0)
    ap.add_argument("--bg_AA", type=float, default=0.0)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--kde_bw", type=float, default=0.02)
    ap.add_argument("--sigma_table", help="CSV (E, sigma) of static standards (e.g. dsDNA): no state may be narrower "
                                           "than the width interpolated at its mean E")
    ap.add_argument("--sigma_floor", type=float, help="constant minimum width of every state (E units)")
    ap.add_argument("--gamma_mode", choices=["per_file", "global"], default="per_file",
                    help="per_file (default): gamma from each file's own APBS regression, folder median if that file's regression is "
                         "not reliable; global: one gamma from all folders")
    ap.add_argument("--width_basis", choices=["dna", "shot"], default="dna",
                    help="dna (default): no state narrower than the static DNA standards (--sens_table); shot: photon shot noise only. "
                         "The other basis is run as a sensitivity fit")
    ap.add_argument("--no_shot_floor", action="store_true",
                    help="do not use the photon-shot-noise lower bound for the state widths (default: use it)")
    ap.add_argument("--dna_dir", help="folder with DNA*_APBS.csv controls (default: data/DNA standarts); only used to COMMENT "
                                      "on alpha/delta, never to correct")
    ap.add_argument("--no_sensitivity", action="store_true", help="skip the second fit with the DNA width table")
    ap.add_argument("--sens_table", default="dna_sigma_floor.csv", help="E,sigma table of the DNA standards for the sensitivity fit")
    ap.add_argument("--no_global_gamma", action="store_true", help="use the median gamma of THIS folder instead of the global one")
    ap.add_argument("--rescan_gamma", action="store_true", help="recompute the global gamma from all data folders")
    ap.add_argument("--allow_unpaired", action="store_true", help="continue although some apbs/dcbs files have no partner")
    ap.add_argument("--R0", type=float, default=51.0)
    ap.add_argument("--R0_err", type=float, default=2.0)
    ap.add_argument("--L_D", type=float, default=20.0)
    ap.add_argument("--L_A", type=float, default=22.0)
    args, _unknown = ap.parse_known_args()
    if args.width_basis == "dna" and not args.sigma_table and args.sigma_floor is None:
        args.sigma_table = args.sens_table
    cf.set_sigma_floor(args.sigma_table, args.sigma_floor if args.sigma_floor is not None else 0.005)
    rng = np.random.default_rng(0)
    bg = (args.bg_DD, args.bg_DA, args.bg_AA)

    folder = Path(args.folder)
    if not folder.is_dir():
        folder = ROOT / "data" / args.folder
    if not folder.is_dir():
        sys.exit(f"ERROR: folder '{args.folder}' not found")
    outdir = Path(args.out) if args.out else ROOT / "results" / folder.name
    outdir.mkdir(parents=True, exist_ok=True)

    pairs, lonely = find_pairs(folder)
    if not pairs:
        sys.exit(f"ERROR: no apbs/dcbs pairs found in {folder}")
    print(f"{len(pairs)} measurement(s) in {folder}")
    if lonely:
        msg = ("  !! UNPAIRED FILES (not analysed): " + ", ".join(lonely) +
               "\n     Rename so that the file names differ only by 'apbs'/'dcbs' (e.g. ..._monomer_dcbs.csv + ..._monomer_apbs.csv).")
        if not args.allow_unpaired:
            sys.exit("ERROR:\n" + msg + "\n     (use --allow_unpaired to continue without them)")
        print(msg)

    # ---------------- pass 1: references
    M = []
    for key, tpath, rpath in pairs:
        try:
            _, rDD, rDA, rAA = read_counts(rpath, bg)
            tdf, tDD, tDA, tAA = read_counts(tpath, bg)
        except Exception as e:
            print(f"  skipping {tpath.name}: {e}")
            continue
        ref = analyse_reference(rDD, rDA, rAA, args, args.n_boot, rng)
        M.append(dict(name=tpath.stem.replace("_dcbs", "").replace("dcbs", "").strip("_"),
                      tpath=tpath, rpath=rpath, tdf=tdf, tDD=tDD, tDA=tDA, tAA=tAA,
                      rDD=rDD, rDA=rDA, rAA=rAA, ref=ref))
    if not M:
        sys.exit("ERROR: nothing to analyse")

    # ---------------- alpha: per file, with an outlier check inside the group of the same laser setting
    def group_of(m):
        return "highS" if "highs" in m["name"].lower() else "normal"
    for m in M:
        m["group"] = group_of(m)
        if m["ref"]["ok"]:
            m["ref"]["alpha_file"] = m["ref"]["alpha"]
            m["alpha_replaced"] = False
    for grp in ("highS", "normal"):
        mem = [m for m in M if m["group"] == grp and m["ref"]["ok"]]
        if len(mem) < 4:
            continue
        av = np.array([m["ref"]["alpha_file"] for m in mem])
        med = float(np.median(av)); mad = 1.4826 * float(np.median(np.abs(av - med))); thr = max(3 * mad, 0.02)
        for m in mem:
            m["alpha_group_median"] = med
            if abs(m["ref"]["alpha_file"] - med) > thr:
                m["ref"]["alpha"] = med
                m["alpha_replaced"] = True
                if len(m["ref"]["boot"]):
                    m["ref"]["boot"][:, 0] = rng.normal(med, max(mad, 0.005), len(m["ref"]["boot"]))

    # ---------------- comments: alpha/delta of each file vs. group median and vs. the DNA controls (informational only)
    dna = []
    dna_dir = Path(args.dna_dir) if args.dna_dir else ROOT / "data" / "DNA standarts"
    if dna_dir.is_dir():
        for p_ in sorted(dna_dir.glob("*.csv")):
            if "apbs" in p_.name.lower():
                try:
                    _, a_, b_, c_ = read_counts(p_, bg)
                    S_ = cf.S_app(a_, b_, c_)
                    dna.append((p_.stem.replace("_APBS", ""), b_[S_ > args.S_donor_only].sum() / a_[S_ > args.S_donor_only].sum(),
                                b_[S_ < args.S_acceptor_only].sum() / c_[S_ < args.S_acceptor_only].sum()))
                except Exception:
                    pass
    cmp_rows = []
    for grp in ("highS", "normal"):
        mem = [m for m in M if m["group"] == grp and m["ref"]["ok"]]
        if mem:
            dmed = float(np.median([m["ref"]["delta"] for m in mem]))
            for m in mem:
                m["delta_group_median"] = dmed
    print("\nCOMMENTS on alpha / delta (value used for the correction = the file's own APBS value; "
          "DNA-control values are from the 2023 session and only shown for comparison)")
    if dna:
        print("  DNA controls: " + "; ".join(f"{n_}: alpha={a_:.3f} delta={d_:.3f}" for n_, a_, d_ in dna))
    for m in M:
        if not m["ref"]["ok"]:
            continue
        r = m["ref"]; a_f = r["alpha_file"]; d_f = r["delta"]
        a_used = r["alpha"]
        da = np.array([x[1] for x in dna]) if dna else np.array([np.nan]); dd = np.array([x[2] for x in dna]) if dna else np.array([np.nan])
        gm = m.get("alpha_group_median", np.nan)
        gm_s = f"{gm:.3f}" if np.isfinite(gm) else "n/a (fewer than 4 files in group)"
        line = (f"  {m['name']}: alpha_file={a_f:.3f} (group {m['group']} median {gm_s}"
                + (f", REPLACED -> {a_used:.3f}" if m.get("alpha_replaced") else "")
                + f"; vs DNA {np.nanmin(da):.3f}-{np.nanmax(da):.3f}: {a_f - np.nanmean(da):+.3f}) | "
                f"delta={d_f:.3f} (group median {m.get('delta_group_median', np.nan):.3f}: {d_f - m.get('delta_group_median', np.nan):+.3f}; "
                f"vs DNA {np.nanmin(dd):.3f}-{np.nanmax(dd):.3f}: {d_f - np.nanmean(dd):+.3f})")
        print(line)
        cmp_rows.append(dict(measurement=m["name"], group=m["group"], alpha_file=a_f, alpha_used=a_used, alpha_replaced=m.get("alpha_replaced", False),
                             alpha_group_median=m.get("alpha_group_median", np.nan), delta=d_f, delta_group_median=m.get("delta_group_median", np.nan),
                             dna_alpha_mean=float(np.nanmean(da)), dna_delta_mean=float(np.nanmean(dd))))
    pd.DataFrame(cmp_rows).to_csv(outdir / "alpha_delta_comparison.csv", index=False)

    # ---------------- gamma: ONE global value (median of all reliable regressions of all constructs), beta per file
    def scan_global_gamma():
        cache = ROOT / "results" / "global_gamma.csv"
        if cache.exists() and not args.rescan_gamma:
            t = pd.read_csv(cache)
        else:
            rr, rows_ = np.random.default_rng(5), []
            for fdir in sorted((ROOT / "data").glob("TF_*/*")):
                if not fdir.is_dir():
                    continue
                prs, _ = find_pairs(fdir)
                for key, tp, rp in prs:
                    try:
                        _, a_, b_, c_ = read_counts(rp, bg)
                        rf = analyse_reference(a_, b_, c_, args, 40, rr)
                    except Exception:
                        continue
                    if rf["ok"] and rf["gamma_reliable"]:
                        rows_.append(dict(construct=fdir.parent.name, state=fdir.name, measurement=tp.stem, gamma=rf["gamma_reg"], gamma_sd=rf["gamma_reg_sd"]))
            t = pd.DataFrame(rows_)
            if len(t):
                cache.parent.mkdir(parents=True, exist_ok=True)
                t.to_csv(cache, index=False)
        return t
    glob_gamma = None
    if args.gamma is None and not args.no_global_gamma and args.gamma_mode in ("global", "per_file"):
        try:
            tg = scan_global_gamma()
            if len(tg) >= 5:
                glob_gamma = (float(tg.gamma.median()), float(tg.gamma.std()), len(tg), tg)
        except Exception as e:
            print(f"  (global gamma scan failed: {e})")

    # ---------------- common gamma from the reliable measurements
    rel = [m for m in M if m["ref"]["ok"] and m["ref"]["gamma_reliable"]]
    if args.gamma is not None:
        gc, gc_sd = args.gamma, args.gamma_err
    elif args.gamma_mode == "global" and glob_gamma is not None:
        gc, gc_sd = glob_gamma[0], glob_gamma[1]
    elif rel and args.gamma_mode == "per_file":
        gc = float(np.median([m["ref"]["gamma_reg"] for m in rel]))
        gc_sd = max(float(np.std([m["ref"]["gamma_reg"] for m in rel])) / np.sqrt(len(rel)), 0.02)
    elif glob_gamma is not None:
        gc, gc_sd = glob_gamma[0], glob_gamma[1]
    elif rel:
        wts = np.array([1 / max(m["ref"]["gamma_reg_sd"], 0.02) ** 2 for m in rel])
        gc = float(np.median([m["ref"]["gamma_reg"] for m in rel]))
        gc_sd = float(np.sqrt(1 / wts.sum()))
        gc_sd = max(gc_sd, float(np.std([m["ref"]["gamma_reg"] for m in rel])) / np.sqrt(len(rel)))
    else:
        gc, gc_sd = 1.0, 0.2
        print("  WARNING: gamma could not be determined from ANY measurement -> gamma = 1 +/- 0.2 assumed.\n"
              "           Pass a measured value with --gamma.")
    print(f"\nCommon gamma = {gc:.2f} +/- {gc_sd:.2f}"
          + ("  (user value)" if args.gamma is not None else
             (f"  (GLOBAL: median of {glob_gamma[2]} reliable regressions over all constructs; +/- = their spread)"
              if (args.gamma_mode == "global" and glob_gamma is not None)
              else f"  (fallback for files without a reliable own regression: median of {len(rel)} of {len(M)} reliable files in this folder)")))
    if args.gamma_mode == "per_file":
        print("  gamma mode: each file uses the gamma of its OWN apbs regression when that is reliable")
    if glob_gamma is not None:
        tg = glob_gamma[3]
        print("  per construct: " + "; ".join(f"{c}: {g.gamma.median():.2f} (n={len(g)})" for c, g in tg.groupby("construct")))

    # ---------------- pass 2: final factors, boot factors, corrected bursts
    for m in M:
        r = m["ref"]
        if not r["ok"]:
            m["factors"] = None
            continue
        if args.gamma is not None:
            g, src = args.gamma, "user"
        elif args.gamma_mode == "global" and glob_gamma is not None:
            g, src = gc, "global"
            if r["gamma_reliable"] and abs(r["gamma_reg"] - gc) > 2 * max(gc_sd, r["gamma_reg_sd"]):
                m["gamma_flag"] = f"own regression gamma {r['gamma_reg']:.2f} deviates from the global {gc:.2f} by more than 2 sigma (global value used)"
        elif r["gamma_reliable"]:
            g, src = r["gamma_reg"], "own"
            if abs(g / gc - 1) > args.gamma_tol:
                m["gamma_flag"] = (f"own gamma {g:.2f} differs by more than {100 * args.gamma_tol:.0f}% from the folder median {gc:.2f} "
                                   f"(own value used, as requested - check this file; --gamma_mode global would use one value)")
            elif glob_gamma is not None and abs(g - glob_gamma[0]) > 2 * max(glob_gamma[1], r["gamma_reg_sd"]):
                m["gamma_flag"] = f"own gamma {g:.2f} is more than 2 sigma from the all-construct median {glob_gamma[0]:.2f} (own value used)"
        else:
            g, src = gc, "common"
            m["gamma_flag"] = f"own gamma regression not reliable (E span {r['span']:.2f}) -> folder median {gc:.2f} used"
        al, de = r["alpha"], r["delta"]
        be = beta_given_gamma(m["rDD"], m["rDA"], m["rAA"], al, de, g, args)
        rows = []
        for (a_, d_, g_), i in zip(r["boot"], r["idx"]):
            gb = g_ if src == "own" else g          # the shared gamma uncertainty is drawn once per merged bootstrap round
            bb = beta_given_gamma(m["rDD"][i], m["rDA"][i], m["rAA"][i], a_, d_, gb, args)
            if np.isfinite(bb):
                rows.append((a_, d_, gb, bb))
        rows = np.array(rows)
        m["factors"] = dict(alpha=al, delta=de, gamma=g, beta=be, gamma_src=src)
        m["boot_rows"] = rows
        m["sd"] = rows.std(axis=0) if len(rows) > 10 else np.full(4, np.nan)
        E, S = cf.correct(m["tDD"], m["tDA"], m["tAA"], al, de, g, be)
        keep = np.isfinite(E) & np.isfinite(S) & (S > args.S_min) & (S < args.S_max) & (E > -0.1) & (E < 1.1)
        m["E_all"], m["S_all"], m["keep"] = E, S, keep
        m["E"] = E[keep]

    good = [m for m in M if m["factors"] is not None]

    # ---------------- compatibility
    for m in M:
        m["reasons"] = []
        if m["factors"] is None:
            m["reasons"].append("reference has too few donor-only/acceptor-only bursts")
        elif len(m["E"]) < args.min_bursts:
            m["reasons"].append(f"only {len(m['E'])} selected bursts")
    cand = [m for m in good if not m["reasons"]]
    forced = [m for m in cand if matches(m["name"], args.include)]
    for m in cand:
        if matches(m["name"], args.exclude):
            m["reasons"].append("excluded by user")
    cand = [m for m in cand if not m["reasons"]]
    while len(cand) > 2:
        w = []
        for m in cand:
            others = np.concatenate([o["E"] for o in cand if o is not m])
            w.append(w1(m["E"], others))
        m["w1_last"] = None
        for m_, w_ in zip(cand, w):
            m_["w1"] = w_
        order = [i for i in np.argsort(w)[::-1] if cand[i] not in forced]
        if not order or w[order[0]] <= args.w1_max:
            break
        worst = cand[order[0]]
        worst["reasons"].append(f"E distribution differs from the others (Wasserstein {w[order[0]]:.3f} > {args.w1_max})")
        cand.remove(worst)
    for m in good:
        if "w1" not in m or m in good and m["reasons"] and m["w1"] is None:
            pass
    pool_all = np.concatenate([m["E"] for m in cand]) if cand else np.array([])
    for m in good:
        others = [o["E"] for o in cand if o is not m]
        m["w1"] = w1(m["E"], np.concatenate(others)) if others and len(m["E"]) > 20 else np.nan
    included = [m for m in M if not m["reasons"]]
    if len(included) == 0:
        sys.exit("ERROR: no compatible measurement left; use --include or relax --w1_max")

    # settings flag (power ratio): delta/beta vs median
    med_d = np.median([m["factors"]["delta"] for m in good])
    med_b = np.median([m["factors"]["beta"] for m in good])
    for m in good:
        f = m["factors"]
        flags = []
        if abs(f["delta"] / med_d - 1) > 0.15:
            flags.append("delta")
        if abs(f["beta"] / med_b - 1) > 0.15:
            flags.append("beta")
        m["settings"] = "different laser power ratio ({})".format("+".join(flags)) if flags else ""

    # ---------------- photon-shot-noise lower bound of the state widths (from the data being fitted)
    if not args.no_shot_floor:
        nph = np.concatenate([(m["tDD"] + m["tDA"])[m["keep"]] for m in included])
        cf.set_shot_floor(float(np.median(nph)))
        print(f"\nWidth lower bound = max( photon shot noise sqrt(E(1-E)/N) with N = {np.median(nph):.0f} photons"
              + (f", DNA-standard table '{args.sigma_table}' )" if args.sigma_table else " )")
              + f"  [basis: {'DNA standards' if args.sigma_table else 'shot noise'}]")

    # ---------------- summary table
    rows = []
    for m in M:
        f = m["factors"]
        row = dict(measurement=m["name"], bursts_dcbs=len(m["tDD"]), ref_bursts=m["ref"]["n_ref"])
        if f:
            sd = m["sd"]
            row.update(alpha=f["alpha"], alpha_sd=sd[0], delta=f["delta"], delta_sd=sd[1],
                       gamma=f["gamma"], gamma_sd=sd[2], gamma_source=f["gamma_src"],
                       gamma_regression=m["ref"]["gamma_reg"], E_span_of_reference=m["ref"]["span"],
                       beta=f["beta"], beta_sd=sd[3], selected_bursts=len(m["E"]))
            pe = cf.point_estimates(m["E"], args.kde_bw) if len(m["E"]) > 20 else dict(mean=np.nan, median=np.nan, mode=np.nan)
            row.update(E_mean=pe["mean"], E_median=pe["median"], E_mode=pe["mode"], W1_to_pool=m.get("w1", np.nan),
                       states=state_string(m["E"], args) if len(m["E"]) > 100 else "", settings=m["settings"])
        row["merged"] = "yes" if not m["reasons"] else "NO"
        row["reason"] = "; ".join(m["reasons"])
        rows.append(row)
    S_df = pd.DataFrame(rows)
    S_df.to_csv(outdir / "measurement_summary.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    show = S_df[[c for c in ("measurement", "selected_bursts", "alpha", "delta", "gamma", "gamma_source", "beta",
                             "E_median", "E_mode", "W1_to_pool", "states", "merged", "reason") if c in S_df.columns]].copy()
    for c in ("alpha", "delta", "gamma", "beta", "E_median", "E_mode", "W1_to_pool"):
        if c in show:
            show[c] = show[c].round(3)
    print("\nPER-MEASUREMENT RESULTS (each corrected with its own reference)\n")
    print(show.to_string(index=False))
    for m in good:
        if m["settings"]:
            print(f"  note: {m['name']}: {m['settings']} (corrected with its own factors)")
    for m in good:
        if m.get("gamma_flag"):
            print(f"  note: {m['name']}: {m['gamma_flag']}")
        if m.get("alpha_replaced"):
            print(f"  note: {m['name']}: alpha {m['ref']['alpha_file']:.3f} is an outlier in its group -> group median {m['ref']['alpha']:.3f} used")
    print(f"\n{len(included)} of {len(M)} measurements are compatible and will be merged.")

    # ---------------- merged fit
    x_all = np.concatenate([m["E"] for m in included])
    if str(args.n).lower() == "auto":
        w, mu, sdv, _ = cf.select_model(x_all, args.max_n, args.min_frac, args.min_sep, args.delta_bic)
    else:
        w, mu, sdv, _ = cf.best_mixture(x_all, int(args.n))
    n = len(mu)
    print(f"\n>>> merged fit: {n} FRET state{'s' if n > 1 else ''} from {len(x_all)} bursts")

    boots_mu, boots_w, boots_pe = [], [], []
    for _ in range(args.n_boot):
        pick = rng.integers(0, len(included), len(included))
        es = []
        g_shared = rng.normal(gc, gc_sd) if args.gamma is None else None
        for k in pick:
            m = included[k]
            row = m["boot_rows"][rng.integers(0, len(m["boot_rows"]))].copy()
            if g_shared is not None and m["factors"]["gamma_src"] != "own":
                row[2] = g_shared
            E, S = cf.correct(m["tDD"], m["tDA"], m["tAA"], *row)
            ok = np.isfinite(E) & np.isfinite(S) & (S > args.S_min) & (S < args.S_max) & (E > -0.1) & (E < 1.1)
            es.append(E[ok])
        e = np.concatenate(es)
        if len(e) > 50:
            fb = cf.em_gmm(e, mu)
            boots_w.append(fb[0]); boots_mu.append(fb[1])
            if n == 1:
                boots_pe.append(cf.point_estimates(e, args.kde_bw))
    bm, bw_ = np.array(boots_mu), np.array(boots_w)
    err = bm.std(axis=0); werr = bw_.std(axis=0)
    names = args.names if args.names and len(args.names) == n else (["single"] if n == 1 else [f"state{i + 1}" for i in range(n)])
    E_rep, E_err = list(mu), list(err)
    extra = {}
    if n == 1:
        pe = cf.point_estimates(x_all, args.kde_bw)
        sd_mode = float(np.std([b["mode"] for b in boots_pe]))
        if sd_mode <= 0.05:
            E_rep, E_err = [pe["mode"]], [sd_mode]
        extra = dict(E_mode=pe["mode"], E_median=pe["median"], E_mean=pe["mean"], E_gauss=mu[0])
    out = pd.DataFrame([dict(state=names[i], E=round(E_rep[i], 4), E_err=round(E_err[i], 4), sigma=round(sdv[i], 3), sigma_floor=round(float(cf.sigma_floor(mu[i])[0]), 3),
                             fraction=round(w[i], 3), fraction_err=round(werr[i], 3), **extra) for i in range(n)])
    print("\n", out.to_string(index=False))
    print(f"\n(E_err: bootstrap over {len(included)} measurements incl. each measurement's correction-factor uncertainty)")
    out.to_csv(outdir / "data.csv", index=False)

    merged = pd.concat([pd.DataFrame(dict(measurement=m["name"], n_DD=m["tDD"][m["keep"]], n_DA=m["tDA"][m["keep"]],
                                         n_AA=m["tAA"][m["keep"]], E_corr=m["E_all"][m["keep"]], S_corr=m["S_all"][m["keep"]]))
                        for m in included], ignore_index=True)
    merged.to_csv(outdir / "merged_corrected_bursts.csv", index=False)

    # ---------------- plots
    cmap = plt.get_cmap("tab20")
    labels = [m["name"] for m in M]
    fig, ax = plt.subplots(2, 2, figsize=(max(9, 0.55 * len(M) + 6), 8), sharex=True)
    for a, (key, lab, idx) in zip(ax.ravel(), [("alpha", "alpha (leakage)", 0), ("delta", "delta (direct excitation)", 1),
                                               ("gamma", "gamma", 2), ("beta", "beta", 3)]):
        for i, m in enumerate(M):
            if m["factors"] is None:
                continue
            col = "tab:blue" if not m["reasons"] else "0.6"
            mk = "o" if m["factors"]["gamma_src"] != "common" or key != "gamma" else "s"
            a.errorbar(i, m["factors"][key], yerr=m["sd"][idx], fmt=mk, color=col, capsize=3)
        vals = [m["factors"][key] for m in good]
        a.axhline(np.median(vals), color="k", ls=":", lw=1)
        a.set_ylabel(lab)
    for a in ax[1]:
        a.set_xticks(range(len(M))); a.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    fig.suptitle("Correction factors of every measurement (blue = merged, grey = excluded, square = common gamma used)")
    plt.tight_layout(); plt.savefig(outdir / "factors_per_measurement.png", dpi=180); plt.close()

    nc = 4; nr = int(np.ceil(len(M) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4 * nc, 2.8 * nr), squeeze=False, sharex=True)
    bins = np.linspace(-0.1, 1.1, 41)
    dens_all, _ = np.histogram(x_all, bins=bins, density=True)
    for a, m in zip(axs.ravel(), M):
        if m["factors"] is not None and len(m["E"]):
            a.hist(m["E"], bins=bins, density=True, color="tab:blue" if not m["reasons"] else "0.6", alpha=0.8)
        a.step(0.5 * (bins[:-1] + bins[1:]), dens_all, color="k", lw=1, where="mid")
        a.set_title(m["name"] + ("" if not m["reasons"] else "  [EXCLUDED]"), fontsize=7)
        a.set_xlim(-0.1, 1.1)
    for a in axs.ravel()[len(M):]:
        a.axis("off")
    fig.supxlabel("corrected E (black line = merged distribution)"); fig.supylabel("density")
    plt.tight_layout(); plt.savefig(outdir / "per_measurement_histograms.png", dpi=170); plt.close()

    xs = np.linspace(-0.1, 1.1, 500)
    bwid = 1.2 / 40
    comp = [len(x_all) * bwid * w[i] * np.exp(-0.5 * ((xs - mu[i]) / sdv[i]) ** 2) / (sdv[i] * np.sqrt(2 * np.pi)) for i in range(n)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    h = h2(ax[0], merged.E_corr, merged.S_corr.clip(-0.2, 1.2), bins=60, range=[[-0.1, 1.1], [0, 1]], cmap="viridis")
    ax[0].set_xlabel("corrected E"); ax[0].set_ylabel("corrected S"); ax[0].set_title(f"merged E-S ({len(included)} measurements)")
    ax[1].hist(x_all, bins=40, range=(-0.1, 1.1), color="0.8")
    ax[1].plot(xs, np.sum(comp, axis=0), "k-")
    for c, nm, e_, p_ in zip(comp, names, mu, w):
        ax[1].plot(xs, c, "--", label=f"{nm}: E={e_:.2f} ({100 * p_:.0f}%)")
    ax[1].legend(fontsize=8); ax[1].set_xlabel("corrected E"); ax[1].set_ylabel("bursts"); ax[1].set_title("merged accurate-E histogram")
    for i, m in enumerate(included):
        d, _ = np.histogram(m["E"], bins=bins, density=True)
        ax[2].plot(0.5 * (bins[:-1] + bins[1:]), d, color=cmap(i % 20), lw=0.9, alpha=0.8)
    ax[2].plot(0.5 * (bins[:-1] + bins[1:]), dens_all, "k", lw=2, label="merged")
    ax[2].set_xlabel("corrected E"); ax[2].set_ylabel("density"); ax[2].legend(); ax[2].set_title("individual measurements")
    plt.tight_layout(); plt.savefig(outdir / "merged_FRET.png", dpi=200); plt.close()


    # ---------------- uncorrected APBS (reference) E_app-S_app plots, one panel per measurement
    fig, axs = plt.subplots(nr, nc, figsize=(4 * nc, 3.4 * nr), squeeze=False)
    for a, m in zip(axs.ravel(), M):
        DDr, DAr, AAr = m["rDD"], m["rDA"], m["rAA"]
        with np.errstate(invalid="ignore", divide="ignore"):
            e_ = DAr / (DDr + DAr); s_ = (DDr + DAr) / (DDr + DAr + AAr)
        ok_ = np.isfinite(e_) & np.isfinite(s_)
        h2(a, e_[ok_], s_[ok_], bins=50, range=[[0, 1], [0, 1]], cmap="viridis", cmin=1)
        a.axhline(args.S_donor_only, color="w", ls=":", lw=0.8); a.axhline(args.S_acceptor_only, color="w", ls=":", lw=0.8)
        f = m["factors"]
        ttl = m["name"] if f is None else f"{m['name']}\nalpha={f['alpha']:.3f} delta={f['delta']:.3f}"
        a.set_title(ttl, fontsize=7); a.set_xlabel("E_app (uncorrected)", fontsize=7); a.set_ylabel("S_app", fontsize=7)
    for a in axs.ravel()[len(M):]:
        a.axis("off")
    fig.suptitle("Uncorrected APBS reference bursts (dotted: donor-only S>%.1f / acceptor-only S<%.1f windows)" % (args.S_donor_only, args.S_acceptor_only))
    plt.tight_layout(); plt.savefig(outdir / "apbs_uncorrected.png", dpi=170); plt.close()

    # uncorrected vs corrected DCBS, merged
    app = np.concatenate([m["tDA"][m["keep"]] / (m["tDD"][m["keep"]] + m["tDA"][m["keep"]]) for m in included])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(app[np.isfinite(app)], bins=40, range=(-0.1, 1.1), color="0.65", label="uncorrected E_app")
    ax[0].hist(x_all, bins=40, range=(-0.1, 1.1), histtype="step", color="tab:red", lw=2, label="corrected E")
    ax[0].set_xlabel("E"); ax[0].set_ylabel("bursts"); ax[0].legend(); ax[0].set_title("merged DCBS data: before / after correction")
    with np.errstate(invalid="ignore", divide="ignore"):
        allr = [(m["rDA"] / (m["rDD"] + m["rDA"]), (m["rDD"] + m["rDA"]) / (m["rDD"] + m["rDA"] + m["rAA"])) for m in M]
    er = np.concatenate([a_[0] for a_ in allr]); sr = np.concatenate([a_[1] for a_ in allr]); okr = np.isfinite(er) & np.isfinite(sr)
    h2(ax[1], er[okr], sr[okr], bins=60, range=[[0, 1], [0, 1]], cmap="viridis", cmin=1)
    ax[1].set_xlabel("E_app (uncorrected)"); ax[1].set_ylabel("S_app"); ax[1].set_title("all APBS reference bursts, uncorrected")
    plt.tight_layout(); plt.savefig(outdir / "uncorrected_vs_corrected.png", dpi=180); plt.close()

    # ---------------- distances (accurate-FRET graph)
    script = HERE / "smfret_residue_distance.py"
    if script.exists():
        r = subprocess.run([sys.executable, str(script), "data.csv", "--R0", str(args.R0), "--R0_err", str(args.R0_err),
                            "--L_D", str(args.L_D), "--L_A", str(args.L_A), "--out", "residue_distances"],
                           cwd=outdir, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print("\n" + r.stdout.strip().encode("ascii", "replace").decode())
        if r.returncode != 0:
            print(r.stderr[-800:])
    # ---------------- model-free headline numbers
    pm = pd.DataFrame([dict(measurement=m["name"], bursts=len(m["E"]), E_mean=float(np.mean(m["E"])),
                            E_median=float(np.median(m["E"])), E_mode=cf.point_estimates(m["E"], args.kde_bw)["mode"]) for m in included])
    pe_all = cf.point_estimates(x_all, args.kde_bw)
    mf = dict(n_measurements=len(included), n_bursts=len(x_all), E_mean_pooled=pe_all["mean"], E_median_pooled=pe_all["median"],
              E_mode_pooled=pe_all["mode"], E_mean_sd_between_measurements=float(pm.E_mean.std()),
              E_mean_sem=float(pm.E_mean.std() / np.sqrt(len(pm))), E_median_sd_between_measurements=float(pm.E_median.std()),
              gamma=gc, gamma_sd=gc_sd,
              E_mean_shift_per_gamma_sd=float(abs(np.mean(cf.correct(np.concatenate([m["tDD"][m["keep"]] for m in included]),
                    np.concatenate([m["tDA"][m["keep"]] for m in included]), np.concatenate([m["tAA"][m["keep"]] for m in included]),
                    np.mean([m["factors"]["alpha"] for m in included]), np.mean([m["factors"]["delta"] for m in included]), gc + gc_sd,
                    np.mean([m["factors"]["beta"] for m in included]))[0]) - np.mean(x_all))))
    pd.DataFrame([mf]).to_csv(outdir / "model_free_summary.csv", index=False)
    pm.to_csv(outdir / "per_measurement_E.csv", index=False)
    print("\nMODEL-FREE RESULT (headline): pooled mean E = {E_mean_pooled:.3f}, median {E_median_pooled:.3f}, mode {E_mode_pooled:.3f}; "
          "spread of the per-measurement means = {E_mean_sd_between_measurements:.3f} (n = {n_measurements}); "
          "gamma +/- 1 SD shifts the mean by about {E_mean_shift_per_gamma_sd:.3f}".format(**mf))

    # ---------------- sensitivity: the same data fitted with the OTHER width basis
    if not args.no_sensitivity:
        try:
            keep_floor = dict(cf._FLOOR)
            main_basis = "DNA standards" if args.sigma_table else "shot noise"
            alt_basis = "shot noise" if args.sigma_table else "DNA standards"
            if args.sigma_table:
                cf._FLOOR["E"] = cf._FLOOR["sigma"] = None
                cf._FLOOR["const"] = 0.005
            else:
                cf.set_sigma_floor(args.sens_table, 0.005)
            if str(args.n).lower() == "auto":
                w2, mu2, sd2, _ = cf.select_model(x_all, args.max_n, args.min_frac, args.min_sep, args.delta_bic, verbose=False)
            else:
                w2, mu2, sd2, _ = cf.best_mixture(x_all, int(args.n))
            sens = outdir / "alt_width_sensitivity"; sens.mkdir(exist_ok=True)
            n2 = len(mu2)
            pd.DataFrame([dict(state=f"state{i + 1}", E=round(mu2[i], 4), E_err=np.nan, sigma=round(sd2[i], 3),
                               sigma_floor=round(float(cf.sigma_floor(mu2[i])[0]), 3), fraction=round(w2[i], 3), fraction_err=np.nan)
                          for i in range(n2)]).to_csv(sens / "data.csv", index=False)
            if script.exists():
                subprocess.run([sys.executable, str(script), "data.csv", "--R0", str(args.R0), "--R0_err", str(args.R0_err),
                                "--L_D", str(args.L_D), "--L_A", str(args.L_A), "--out", "residue_distances"],
                               cwd=sens, capture_output=True, text=True, encoding="utf-8", errors="replace")
            cf._FLOOR.update(keep_floor)
            print(f"\nSENSITIVITY: same bursts, widths bounded by {alt_basis} instead of {main_basis} (the headline result uses {main_basis})")
            comp = []
            for i in range(n):
                j = int(np.argmin(np.abs(mu2 - mu[i])))
                comp.append(dict(state=names[i], E_main=mu[i], E_alt=mu2[j], dE=mu2[j] - mu[i], pop_main=w[i], pop_alt=w2[j],
                                 dpop=w2[j] - w[i], sigma_main=sdv[i], sigma_alt=sd2[j], main_basis=main_basis, alt_basis=alt_basis))
            comp = pd.DataFrame(comp)
            comp.to_csv(outdir / "state_comparison_main_vs_alt_width.csv", index=False)
            print(comp.drop(columns=["main_basis", "alt_basis"]).round(3).to_string(index=False))
            print(f"  number of states: main {n}, alternative {n2}")
            flags = []
            for i in range(n):
                fl = float(cf.sigma_floor(mu[i])[0])
                if sdv[i] <= 1.1 * fl:
                    flags.append(f"{names[i]}: width sits at the lower bound ({fl:.3f}) - the data may be narrower than the bound")
                if w[i] < 0.15:
                    flags.append(f"{names[i]}: small population ({100 * w[i]:.0f}%)")
                if abs(comp.dE.iloc[i]) > 0.05:
                    flags.append(f"{names[i]}: E moves by {comp.dE.iloc[i]:+.2f} with the {alt_basis} width basis")
                if abs(comp.dpop.iloc[i]) > 0.10:
                    flags.append(f"{names[i]}: population moves by {100 * comp.dpop.iloc[i]:+.0f} points with the {alt_basis} width basis")
                if mu[i] < 0.15 or mu[i] > 0.85:
                    flags.append(f"{names[i]}: E = {mu[i]:.2f} outside the 0.15-0.85 range, distance only qualitative")
            if n != n2:
                flags.append(f"number of states changes from {n} to {n2} with the {alt_basis} width basis")
            print("STATE RELIABILITY FLAGS: " + ("none" if not flags else ""))
            for f_ in flags:
                print("  - " + f_)
            (outdir / "state_flags.txt").write_text("\n".join(flags) if flags else "none")
        except Exception as e:
            print(f"  sensitivity fit failed: {e}")

    print(f"\nAll results are in: {outdir}")


if __name__ == "__main__":
    main()

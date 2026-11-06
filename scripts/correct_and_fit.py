#!/usr/bin/env python3
"""
Step 1: derive ALL correction factors from your own all-photon-burst-search (APBS) file,
correct your burst data, and find the FRET state(s) -- with the NUMBER OF STATES CHOSEN
AUTOMATICALLY (1, 2 or 3 Gaussians).

Corrections (reference = APBS file with donor-only, acceptor-only and FRET bursts):
   donor-only      (S_app ~ 1)  -> leakage   alpha = sum(n_DA) / sum(n_DD)
   acceptor-only   (S_app ~ 0)  -> direct excitation   delta = sum(n_DA) / sum(n_AA)
   FRET bursts                  -> gamma, beta from  Y = a + b*P  (gamma = a/(a+b), beta = a+b)
        P = DA'/(DD+DA'),  Y = AA/(DD+DA'),  DA' = DA - alpha*DD - delta*AA
   E = DA' / (DA' + gamma*DD)
   S = 1 / (1 + (1/beta) * AA / (gamma*DD + DA'))

Number of states (default --n auto):
   Gaussian mixtures with 1..max_n components are fitted to the corrected E values
   (maximum likelihood, EM algorithm). A model with more components is accepted only if
     - the BIC improves by more than --delta_bic (default 10 = strong evidence),
     - every component holds at least --min_frac of the bursts (default 0.10), and
     - neighbouring component means are at least --min_sep apart in E (default 0.15).
   Otherwise the simpler model is kept. Use  --n 2  (or --centers) to force a fixed number.

Uncertainty: the whole procedure (alpha, delta, gamma, beta, S window, mixture fit) is repeated
with the reference bursts and the target PTU files bootstrapped, so E_err includes the
uncertainty of the correction factors.

Output: data.csv (state,E,E_err,sigma,fraction) -> smfret_residue_distance.py data.csv

Several CSV files for ONE state (replicates, several measurement days, ...): just list them; wildcards work too.
  python correct_and_fit.py rep1.csv rep2.csv rep3.csv --ref APBS.csv --n auto
  python correct_and_fit.py "rep*.csv" --ref APBS_1.csv APBS_2.csv
All target files are pooled (same correction factors for all) and the error bars come from resampling the
individual measurement files. A table lists each file separately so you can check that they agree.

Usage
  python correct_and_fit.py Target.csv --ref APBS.csv                 # automatic 1/2/3 states
  python correct_and_fit.py Target.csv --ref APBS.csv --n 2 --names low high   # force 2 states
  python correct_and_fit.py Target.csv --ref APBS.csv --max_n 2 --gamma 1.1 --gamma_err 0.05

IMPORTANT: correction factors are only valid for data measured with identical laser powers,
filters, detectors and alignment as the reference measurement.
Requirements: numpy pandas matplotlib
"""
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2D histograms in the standard "viridis" colour code (dark purple = empty, yellow = most frequent)
def h2(ax, x, y, **kw):
    kw.pop("cmap", None); kw.pop("cmin", None)
    ax.set_facecolor(plt.get_cmap("viridis")(0.0))
    return ax.hist2d(x, y, cmap="viridis", **kw)


# ---------------------------------------------------------------- helpers
def read_table(path):
    """Read a CSV in either format: comma/dot (international) or semicolon/decimal-comma (European Excel)."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        first = fh.readline()
    if first.count(";") > first.count(","):
        return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    return pd.read_csv(path, encoding="utf-8-sig")


def load_many(specs, label):
    """Read one or several CSV files (wildcards allowed, e.g. "rep*.csv") and stack them.
    A column 'source_csv' records which file each burst came from."""
    import glob
    from pathlib import Path
    paths = []
    for sp in ([specs] if isinstance(specs, str) else list(specs)):
        hits = sorted(glob.glob(sp)) if any(c in sp for c in "*?[") else [sp]
        if not hits:
            raise SystemExit(f"ERROR: no file matches '{sp}' ({label})")
        paths += hits
    frames = []
    need = ["n_DD", "n_DA", "n_AA", "time_length"]
    for pth in paths:
        if not Path(pth).exists():
            raise SystemExit(f"ERROR: file not found: {pth} ({label})")
        df = read_table(pth)
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise SystemExit(f"ERROR: {pth} lacks the columns {miss} (found: {list(df.columns)[:10]} ...)")
        if "filename" not in df.columns:
            df["filename"] = Path(pth).stem
        df["source_csv"] = Path(pth).name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if len(paths) > 1:
        print(f"{label}: {len(paths)} CSV files, {len(out)} bursts in total")
        for pth, f in zip(paths, frames):
            print(f"   {Path(pth).name}: {len(f)} bursts")
    return out


def counts(df, bg):
    T = df.time_length.to_numpy(float)
    return (df.n_DD.to_numpy(float) - bg[0] * T,
            df.n_DA.to_numpy(float) - bg[1] * T,
            df.n_AA.to_numpy(float) - bg[2] * T)


def S_app(DD, DA, AA):
    return (DD + DA) / (DD + DA + AA)


def derive_factors(DD, DA, AA, args, fixed_gamma=None):
    """alpha, delta, gamma, beta from one set of reference bursts."""
    S = S_app(DD, DA, AA)
    d = S > args.S_donor_only
    a = S < args.S_acceptor_only
    alpha = DA[d].sum() / DD[d].sum()
    delta = DA[a].sum() / AA[a].sum()

    f = (S > args.S_fret_lo) & (S < args.S_fret_hi)
    dd, aa = DD[f], AA[f]
    dac = DA[f] - alpha * dd - delta * aa
    ok = dac > 0
    dd, dac, aa = dd[ok], dac[ok], aa[ok]
    P = dac / (dd + dac)
    edges = np.linspace(0.05, 1.0, 10)
    xs, ys, ws = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (P >= lo) & (P < hi)
        if m.sum() > 12:
            xs.append(P[m].mean())
            ys.append(aa[m].sum() / (dd[m] + dac[m]).sum())
            ws.append(m.sum())
    gamma, beta = 1.0, 1.0
    if len(xs) >= 4:
        b, a_ = np.polyfit(xs, ys, 1, w=np.sqrt(ws))
        if (a_ + b) > 0 and a_ > 0:
            gamma, beta = a_ / (a_ + b), a_ + b
    if fixed_gamma is not None:
        gamma = fixed_gamma
    return alpha, delta, gamma, beta


def fret_lever_arm(DD, DA, AA, alpha, delta, args):
    """E_PR range (5-95 %) of the FRET bursts used for the gamma/beta regression. The regression of
    1/S against E_PR (Lee et al. 2005, Eq. 20) only works if these bursts cover a wide range of E."""
    S = S_app(DD, DA, AA)
    f = (S > args.S_fret_lo) & (S < args.S_fret_hi)
    dac = DA[f] - alpha * DD[f] - delta * AA[f]
    ok = dac > 0
    P = dac[ok] / (DD[f][ok] + dac[ok])
    return (np.percentile(P, 95) - np.percentile(P, 5)) if len(P) > 20 else 0.0


def correct(DD, DA, AA, alpha, delta, gamma, beta):
    DAc = DA - alpha * DD - delta * AA
    E = DAc / (DAc + gamma * DD)
    S = 1.0 / (1.0 + (1.0 / beta) * AA / (gamma * DD + DAc))
    return E, S


# ------------------------------------------------------ minimum Gaussian width
# A state can never be narrower than a STATIC sample (e.g. dsDNA standards) measured with the same burst selection:
# its width is photon noise + instrument broadening. The floor is either a constant or a table E -> sigma
# (linearly interpolated, constant outside the table range).
_FLOOR = {"E": None, "sigma": None, "const": 0.02, "shot_N": None}


def set_sigma_floor(table_path=None, const=None):
    if table_path:
        from pathlib import Path as _P
        here = _P(__file__).resolve().parent
        cands = [_P(table_path), here.parent / table_path, here.parent / "data" / table_path, here / table_path]
        found = next((c for c in cands if c.is_file()), None)
        if found is None and _P(table_path).name.lower() == "dna_sigma_floor.csv":
            # built-in copy of the DNA-standard widths (DCBS apparent E, S 0.4-0.8, >=200 photons) so the run never fails on a missing file
            print("NOTE: data/dna_sigma_floor.csv not found -> built-in DNA-standard widths used "
                  "(E 0.287 -> 0.077, 0.545 -> 0.100, 0.747 -> 0.110). Put the file into data/ to use your own table.")
            _FLOOR["E"], _FLOOR["sigma"] = np.array([0.287, 0.545, 0.747]), np.array([0.077, 0.100, 0.110])
        elif found is None:
            raise SystemExit(f"ERROR: --sigma_table '{table_path}' not found (looked in the current folder, the project folder and data/)")
        else:
            t = read_table(found)
            t = t.sort_values(t.columns[0])
            _FLOOR["E"], _FLOOR["sigma"] = t.iloc[:, 0].to_numpy(float), t.iloc[:, 1].to_numpy(float)
    if const is not None:
        _FLOOR["const"] = float(const)


def set_shot_floor(n_photons):
    """Physical lower bound of a state width: photon shot noise sqrt(E(1-E)/N), N = typical photons (DD+DA') per burst
    of the data being fitted. Computed from the data itself, not from a reference sample."""
    _FLOOR["shot_N"] = float(n_photons) if n_photons else None


def sigma_floor(mu):
    """Minimum width at mean E. Inside the table range: linear interpolation. Outside it the table value of the
    nearest end is scaled by sqrt(E(1-E)) (the width of a binomial/photon-noise distribution shrinks towards E = 0 and 1),
    so the floor is NOT extrapolated as a constant."""
    mu = np.atleast_1d(np.asarray(mu, float))
    base = np.full(mu.shape, _FLOOR["const"])
    if _FLOOR["E"] is not None:
        E, sg = _FLOOR["E"], _FLOOR["sigma"]
        f = lambda x: np.sqrt(np.clip(x * (1 - x), 1e-6, None))
        tab = np.interp(mu, E, sg)
        lo, hi = mu < E[0], mu > E[-1]
        tab[lo] = sg[0] * f(mu[lo]) / f(E[0])
        tab[hi] = sg[-1] * f(mu[hi]) / f(E[-1])
        base = np.maximum(base, tab)
    if _FLOOR["shot_N"]:
        base = np.maximum(base, np.sqrt(np.clip(mu * (1 - mu), 1e-4, None) / _FLOOR["shot_N"]))
    return base


# ------------------------------------------------------ Gaussian mixture (EM)
def em_gmm(x, means0, n_iter=500, tol=1e-9, min_sigma=0.005):
    """Maximum-likelihood Gaussian mixture. Returns weights, means, sigmas, logL (sorted by mean)."""
    x = np.asarray(x, float)
    N, k = len(x), len(means0)
    mu = np.array(means0, float)
    sd = np.full(k, max(x.std() / k, 0.05))
    w = np.full(k, 1.0 / k)
    ll_old = -np.inf
    for _ in range(n_iter):
        pdf = w * np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        tot = pdf.sum(1) + 1e-300
        ll = np.log(tot).sum()
        r = pdf / tot[:, None]
        nk = r.sum(0) + 1e-12
        w = nk / N
        mu = (r * x[:, None]).sum(0) / nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk)
        sd = np.maximum(sd, np.maximum(min_sigma, sigma_floor(mu)))
        if abs(ll - ll_old) < tol * abs(ll):
            break
        ll_old = ll
    o = np.argsort(mu)
    return w[o], mu[o], sd[o], ll


def best_mixture(x, k, n_restarts=8, seed=0):
    """Best of several initialisations (quantile-based + random) for k components."""
    rng = np.random.default_rng(seed)
    inits = [np.quantile(x, [(i + 0.5) / k for i in range(k)])]
    inits += [np.sort(rng.choice(x, k, replace=False)) for _ in range(n_restarts)]
    fits = [em_gmm(x, m0) for m0 in inits]
    return max(fits, key=lambda f: f[3])


def bic(ll, k, N):
    return -2 * ll + (3 * k - 1) * np.log(N)


def valid(fit, min_frac, min_sep):
    w, mu, sd, ll = fit
    if np.any(w < min_frac):
        return False
    if len(mu) > 1 and np.any(np.diff(mu) < min_sep):
        return False
    return True


def select_model(x, max_n, min_frac, min_sep, delta_bic, verbose=True):
    """Choose the simplest mixture that is clearly better than the previous one.
    A fit with a tiny or overlapping component is skipped (a larger k is still tried);
    the search stops when a valid larger model no longer improves the BIC enough."""
    N = len(x)
    best = best_mixture(x, 1)
    best_bic = bic(best[3], 1, N)
    if verbose:
        print("\nModel selection (Gaussian mixtures fitted to corrected E):")
        print(f"  k=1  logL={best[3]:9.1f}  BIC={best_bic:9.1f}  sigma={best[2][0]:.3f}  -> baseline")
    for k in range(2, max_n + 1):
        f = best_mixture(x, k)
        b = bic(f[3], k, N)
        ok = valid(f, min_frac, min_sep)
        gain = best_bic - b
        desc = "means " + ", ".join(f"{m:.2f}" for m in f[1]) + " | fractions " + ", ".join(f"{v:.2f}" for v in f[0])
        if not ok:
            if verbose:
                print(f"  k={k}  BIC={b:9.1f}  ({desc}) -> skipped: component too small (<{min_frac}) or too close (<{min_sep} in E)")
            continue
        accept = gain > delta_bic
        if verbose:
            print(f"  k={k}  BIC={b:9.1f}  (gain {gain:7.1f}; {desc}) -> "
                  + ("accepted" if accept else f"rejected: BIC gain below {delta_bic}"))
        if accept:
            best, best_bic = f, b
        else:
            break
    return best


# ------------------------------------------------- model-free point estimates
def point_estimates(x, bw=0.02):
    """Model-free estimates of the central E: peak of a smoothed histogram (mode), median, mean."""
    step = 0.002
    edges = np.arange(-0.1, 1.1 + step, step)
    cnt, _ = np.histogram(x, bins=edges)
    cen = 0.5 * (edges[:-1] + edges[1:])
    kk = np.arange(-int(4 * bw / step), int(4 * bw / step) + 1) * step
    dens = np.convolve(cnt, np.exp(-0.5 * (kk / bw) ** 2), mode="same")
    return dict(mode=float(cen[dens.argmax()]), median=float(np.median(x)), mean=float(np.mean(x)))


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="+", help="burst CSV file(s) to analyse; several files = one state (pooled)")
    ap.add_argument("--ref", nargs="+", help="APBS CSV file(s) with donor-only/acceptor-only/FRET bursts "
                                  "(default: the target file itself)")
    ap.add_argument("--n", default="auto", help="'auto' (default) or a fixed number of states")
    ap.add_argument("--max_n", type=int, default=3, help="largest number of states tried in auto mode")
    ap.add_argument("--min_frac", type=float, default=0.10, help="minimum fraction of bursts per state")
    ap.add_argument("--min_sep", type=float, default=0.15, help="minimum E separation between states")
    ap.add_argument("--delta_bic", type=float, default=10.0, help="BIC improvement required for an extra state")
    ap.add_argument("--centers", nargs="+", type=float, help="fix the number of states and give start values")
    ap.add_argument("--names", nargs="+", help="state names, low E -> high E (used if the count matches)")
    ap.add_argument("--estimator", default="auto", choices=["auto", "gauss", "mode", "median", "mean"],
                    help="how E is reported for a SINGLE state: mode = peak of smoothed histogram, "
                         "gauss = Gaussian-fit mean, median, mean. auto = mode (multi-state fits always use "
                         "the mixture means)")
    ap.add_argument("--kde_bw", type=float, default=0.02, help="smoothing width (in E) for the mode estimate")
    ap.add_argument("--sigma_table", help="CSV (E, sigma) of static standards (e.g. dsDNA): no state may be narrower "
                                           "than the width interpolated at its mean E")
    ap.add_argument("--sigma_floor", type=float, help="constant minimum width of every state (E units)")
    ap.add_argument("--gamma", type=float, help="fix gamma instead of estimating it")
    ap.add_argument("--gamma_err", type=float, default=0.0, help="uncertainty on a fixed gamma")
    ap.add_argument("--S_donor_only", type=float, default=0.9)
    ap.add_argument("--S_acceptor_only", type=float, default=0.2)
    ap.add_argument("--S_fret_lo", type=float, default=0.2)
    ap.add_argument("--S_fret_hi", type=float, default=0.8)
    ap.add_argument("--S_min", type=float, default=0.3, help="corrected-S window for the analysed bursts")
    ap.add_argument("--S_max", type=float, default=0.7)
    ap.add_argument("--bg_DD", type=float, default=0.0, help="background rate counts/s")
    ap.add_argument("--bg_DA", type=float, default=0.0)
    ap.add_argument("--bg_AA", type=float, default=0.0)
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--out", default="data.csv")
    args = ap.parse_args()
    set_sigma_floor(args.sigma_table, args.sigma_floor)
    if args.sigma_table or args.sigma_floor:
        print(f"Minimum state width: {'table ' + args.sigma_table if args.sigma_table else ''} {'constant ' + str(args.sigma_floor) if args.sigma_floor else ''}")
    bg = (args.bg_DD, args.bg_DA, args.bg_AA)
    rng = np.random.default_rng(0)

    tgt = load_many(args.target, "target")
    ref = load_many(args.ref, "reference") if args.ref else tgt
    rDD, rDA, rAA = counts(ref, bg)
    S_ref = S_app(rDD, rDA, rAA)
    n_do, n_ao = int((S_ref > args.S_donor_only).sum()), int((S_ref < args.S_acceptor_only).sum())
    if n_do < 20 or n_ao < 20:
        sys.exit(f"ERROR: the reference contains only {n_do} donor-only (S_app > {args.S_donor_only}) and {n_ao} "
                 f"acceptor-only (S_app < {args.S_acceptor_only}) bursts - too few to determine leakage and direct "
                 f"excitation.\nUse an APBS file (all bursts, no S filtering) as reference (--ref).")

    # ---- nominal correction factors + bootstrap
    alpha, delta, gamma, beta = derive_factors(rDD, rDA, rAA, args, args.gamma)
    n_ref = len(ref)
    par_boot = []
    for _ in range(args.n_boot):
        i = rng.integers(0, n_ref, n_ref)
        al, de, ga, be = derive_factors(rDD[i], rDA[i], rAA[i], args, args.gamma)
        if args.gamma is not None and args.gamma_err > 0:
            ga = rng.normal(args.gamma, args.gamma_err)
        par_boot.append((al, de, ga, be))
    par_boot = np.array(par_boot)
    sd = par_boot.std(axis=0)
    print("\nCorrection factors from reference (value +/- bootstrap SD):")
    print(f"  alpha (leakage)            = {alpha:.3f} +/- {sd[0]:.3f}")
    print(f"  delta (direct excitation)  = {delta:.3f} +/- {sd[1]:.3f}")
    print(f"  gamma                      = {gamma:.2f} +/- {sd[2]:.2f}"
          + ("  (fixed by user)" if args.gamma is not None else ""))
    print(f"  beta                       = {beta:.2f} +/- {sd[3]:.2f}")
    if args.gamma is None:
        span = fret_lever_arm(rDD, rDA, rAA, alpha, delta, args)
        if gamma == 1.0 and beta == 1.0:
            print("  WARNING: the gamma/beta regression FAILED (too few FRET bursts or too narrow an E range);\n"
                  "           gamma = beta = 1 were ASSUMED. Measure gamma with >= 2 D-A species of different E\n"
                  "           (Lee et al. 2005) and pass it with --gamma.")
        elif span < 0.4:
            print(f"  WARNING: the FRET bursts of the reference cover only a narrow E range (E_PR span {span:.2f} < 0.4)\n"
                  "           -> gamma and beta are unreliable. Use a reference with >= 2 D-A species of clearly\n"
                  "           different E (Lee et al. 2005) or pass a measured --gamma.")
    if args.gamma is None and sd[2] > 0.2:
        print("  WARNING: gamma is poorly determined from this data alone -> its error is large;"
              " use a dedicated gamma measurement and pass --gamma if you have one.")

    # ---- nominal correction of target
    DD, DA, AA = counts(tgt, bg)
    E, S = correct(DD, DA, AA, alpha, delta, gamma, beta)
    tgt = tgt.assign(E_corr=E, S_corr=S)
    keep = (S > args.S_min) & (S < args.S_max) & (E > -0.1) & (E < 1.1)
    d = tgt[keep]
    x_all = d.E_corr.to_numpy()
    print(f"\n{len(tgt)} bursts -> {len(d)} after corrected-S window {args.S_min}-{args.S_max}")
    print(f"mean E_app = {d.E_app.mean():.3f}   mean E_corr = {x_all.mean():.3f}")
    d.to_csv("bursts_corrected.csv", index=False)
    if tgt.source_csv.nunique() > 1:
        rows_pc = []
        for name_, g in d.groupby("source_csv", sort=False):
            pe_ = point_estimates(g.E_corr.to_numpy(), args.kde_bw)
            rows_pc.append(dict(csv_file=name_, bursts=len(g), E_mean=round(pe_["mean"], 3),
                                E_median=round(pe_["median"], 3), E_mode=round(pe_["mode"], 3)))
        pc = pd.DataFrame(rows_pc)
        print("\nPer-file check (corrected E of the selected bursts):")
        print(pc.to_string(index=False))
        spread = pc.E_median.max() - pc.E_median.min()
        print(f"   spread of the file medians: {spread:.3f}")
        if spread > 0.05:
            print("   WARNING: the files differ by more than 0.05 in E -> are they really the same state and the\n"
                  "            same measurement conditions? (different samples/sessions need their own reference)")
        pc.to_csv("per_csv_summary.csv", index=False)

    # ---- choose the model
    if args.centers:
        w, mu, sdv, ll = em_gmm(x_all, args.centers)
        print(f"\nFixed number of states: {len(mu)} (from --centers)")
    elif str(args.n).lower() != "auto":
        k = int(args.n)
        w, mu, sdv, ll = best_mixture(x_all, k)
        print(f"\nFixed number of states: {k}")
    else:
        w, mu, sdv, ll = select_model(x_all, args.max_n, args.min_frac, args.min_sep, args.delta_bic)
    n = len(mu)
    print(f"\n>>> {n} FRET state{'s' if n > 1 else ''} used")
    for i in range(n):
        if sdv[i] > 0.15:
            print(f"NOTE: state at E = {mu[i]:.2f} is broad (sigma = {sdv[i]:.2f}; shot noise alone gives about 0.05-0.10).\n"
                  f"      It probably contains several conformations or dynamics -> its distance is an average.")

    # ---- Monte-Carlo: resample PTU files AND correction factors, redo selection + fit (k fixed)
    tgt["_unit"] = tgt.source_csv.astype(str) + "|" + tgt.filename.astype(str)
    files = tgt["_unit"].unique()
    mask_f = {f: (tgt["_unit"] == f).values for f in files}
    arr = {f: (DD[mask_f[f]], DA[mask_f[f]], AA[mask_f[f]]) for f in files}
    boots, boots_w, boots_pe = [], [], []
    for k_ in range(args.n_boot):
        al, de, ga, be = par_boot[k_]
        pick = rng.choice(files, len(files), replace=True) if len(files) >= 3 else files
        dd = np.concatenate([arr[f][0] for f in pick])
        da = np.concatenate([arr[f][1] for f in pick])
        aa = np.concatenate([arr[f][2] for f in pick])
        e, s_ = correct(dd, da, aa, al, de, ga, be)
        m = (s_ > args.S_min) & (s_ < args.S_max) & (e > -0.1) & (e < 1.1)
        if m.sum() > 20:
            fit_b = em_gmm(e[m], mu)
            boots_w.append(fit_b[0])
            boots.append(fit_b[1])
            if n == 1:
                boots_pe.append(point_estimates(e[m], args.kde_bw))
    boots, boots_w = np.array(boots), np.array(boots_w)
    err = boots.std(axis=0) if len(boots) > 10 else np.full(n, np.nan)
    w_err = boots_w.std(axis=0) if len(boots_w) > 10 else np.full(n, np.nan)

    if args.names and len(args.names) == n:
        names = args.names
    else:
        if args.names:
            print(f"NOTE: {len(args.names)} names given but {n} state(s) found -> using default names")
        names = ["single"] if n == 1 else [f"state{i+1}" for i in range(n)]

    E_report, E_err_report = list(mu), list(err)
    extra_cols = {}
    est_used = "gauss (mixture mean)"
    if n == 1:
        pe = point_estimates(x_all, args.kde_bw)
        pe["gauss"] = float(mu[0])
        pe_err = {k: float(np.std([b[k] for b in boots_pe])) if len(boots_pe) > 10 else np.nan
                  for k in ("mode", "median", "mean")}
        pe_err["gauss"] = float(err[0])
        if args.estimator == "auto":
            if pe_err["mode"] <= 0.05:
                est = "mode"
            else:
                est = "gauss"
                print(f"\nNOTE: the peak position is unstable (bootstrap SD {pe_err['mode']:.2f} > 0.05) because the "
                      "distribution is broad or has several maxima -> using the Gaussian mean instead.")
        else:
            est = args.estimator
        E_report, E_err_report = [pe[est]], [pe_err[est]]
        est_used = est
        print("\nSingle state - comparison of E estimators (value +/- bootstrap SD):")
        for k in ("mode", "median", "mean", "gauss"):
            print(f"   {k:7s} {pe[k]:.3f} +/- {pe_err[k]:.3f}" + ("   <-- used" if k == est else ""))
        extra_cols = {f"E_{k}": round(pe[k], 4) for k in ("mode", "median", "mean", "gauss")}
        if abs(pe["mode"] - pe["mean"]) > 0.03:
            print("   NOTE: mode and mean differ by more than 0.03 -> asymmetric distribution (tail).\n"
                  "         The mode is the position of the main population; the mean is pulled by the tail.")
    rows = []
    for i in range(n):
        row = dict(state=names[i], E=round(E_report[i], 4), E_err=round(E_err_report[i], 4),
                   sigma=round(sdv[i], 3), sigma_floor=round(float(sigma_floor(mu[i])[0]), 3), fraction=round(w[i], 3),
                   fraction_err=round(float(w_err[i]), 3))
        row.update(extra_cols)
        rows.append(row)
    out = pd.DataFrame(rows)
    print("\n", out.to_string(index=False))
    print(f"\n(E = {est_used}; E_err = SD over {len(boots)} bootstrap repeats of PTU files + correction factors)")
    out.to_csv(args.out, index=False)
    print(f"Saved {args.out}  ->  python smfret_residue_distance.py {args.out}")

    # ---- plot
    xs = np.linspace(-0.1, 1.1, 500)
    bins = 40
    bw = 1.2 / bins
    comp = [len(x_all) * bw * w[i] * np.exp(-0.5 * ((xs - mu[i]) / sdv[i]) ** 2)
            / (sdv[i] * np.sqrt(2 * np.pi)) for i in range(n)]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    _e = ref.n_DA / (ref.n_DD + ref.n_DA)          # recomputed from counts (E_app/S_app columns can contain NaN)
    _s = (ref.n_DD + ref.n_DA) / (ref.n_DD + ref.n_DA + ref.n_AA)
    _ok = np.isfinite(_e) & np.isfinite(_s)
    h2(ax[0], _e[_ok], _s[_ok], bins=50, range=[[0, 1], [0, 1]], cmap="viridis")
    ax[0].set_title("reference (apparent)")
    ax[0].set_xlabel("E_app")
    ax[0].set_ylabel("S_app")
    _ok2 = np.isfinite(tgt.E_corr) & np.isfinite(tgt.S_corr)
    h2(ax[1], tgt.E_corr[_ok2], tgt.S_corr[_ok2].clip(-0.2, 1.2), bins=50,
                 range=[[-0.1, 1.1], [-0.2, 1.2]], cmap="viridis")
    ax[1].set_title("target (corrected)")
    ax[1].set_xlabel("E")
    ax[1].set_ylabel("S")
    ax[2].hist(x_all, bins=bins, range=(-0.1, 1.1), color="0.8")
    ax[2].plot(xs, np.sum(comp, axis=0), "k-")
    for c in comp:
        ax[2].plot(xs, c, "--")
    if n == 1:
        ax[2].axvline(pe["mode"], color="r", lw=1.2, label=f"mode {pe['mode']:.2f}")
        ax[2].axvline(pe["mean"], color="b", lw=1.2, ls=":", label=f"mean {pe['mean']:.2f}")
        ax[2].legend(fontsize=8)
    ax[2].set_title(f"{n} state{'s' if n > 1 else ''} (automatic)" if str(args.n).lower() == "auto"
                    else f"{n} state(s) (fixed)")
    ax[2].set_xlabel("corrected E")
    ax[2].set_ylabel("bursts")
    plt.tight_layout()
    plt.savefig("burst_analysis.png", dpi=200)


if __name__ == "__main__":
    main()

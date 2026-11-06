#!/usr/bin/env python3
"""
validate_dna.py - validate the correction chain on the dsDNA standards (DNA5/DNA11/DNA20).

For each standard (one static FRET species, known base-pair separation):
  1. alpha, delta  from its own APBS file (donor-only S>0.9, acceptor-only S<0.2), bootstrap error
  2. gamma, beta   from a GLOBAL regression over the three standards (the Lee 2005 ladder: three
                   species of different E give the lever arm a single species lacks)
  3. corrected E   (ratio of sums, plus median and mode), width vs shot-noise expectation
  4. prediction    accessible-volume (AV) model on B-DNA: dye clouds of radius L_D / L_A around the
                   attachment points, excluded from the DNA cylinder, <E> averaged over all dye pairs,
                   one common angular offset phi fitted to the three standards.
  5. verdict       does the chain reproduce the ladder (E decreasing with bp, AV residual <= tol)?

Usage:  python validate_dna.py --folder "data/DNA standarts" --bp 5 11 20 --out results/DNA_validation
Only for your own analysis.
"""
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    df = pd.read_csv(path)
    for c in ("n_DD", "n_DA", "n_AA"):
        df = df[np.isfinite(df[c])]
    return df["n_DD"].to_numpy(float), df["n_DA"].to_numpy(float), df["n_AA"].to_numpy(float)


def S_app(DD, DA, AA):
    with np.errstate(all="ignore"):
        return (DD + DA) / (DD + DA + AA)


def leak_direct(DD, DA, AA, hi=0.9, lo=0.2):
    S = S_app(DD, DA, AA)
    return DA[S > hi].sum() / DD[S > hi].sum(), DA[S < lo].sum() / AA[S < lo].sum()


def species_point(DD, DA, AA, al, de, lo, hi):
    S = S_app(DD, DA, AA)
    f = (S > lo) & (S < hi)
    dac = DA[f] - al * DD[f] - de * AA[f]
    den = dac + DD[f]
    return dac.sum() / den.sum(), AA[f].sum() / den.sum()    # P (uncorrected-gamma E), y


def global_gamma(points):
    P, y = np.array(points).T
    b, a = np.polyfit(P, y, 1)
    return a / (a + b), a + b, a, b


def corrected(DD, DA, AA, al, de, ga, be, lo, hi):
    S = S_app(DD, DA, AA)
    f = (S > lo) & (S < hi)
    dac = DA[f] - al * DD[f] - de * AA[f]
    E = dac / (dac + ga * DD[f])
    Eroll = dac.sum() / (dac.sum() + ga * DD[f].sum())
    return E, Eroll, f


def shot_sigma(E, n):
    return np.sqrt(np.mean(E) * (1 - np.mean(E)) / np.median(n))


def av_E(dbp, phi_deg, L_D, L_A, R0, rng, N=6000, rise=3.4, twist=34.3, r_att=9.0, r_cyl=10.0):
    def cloud(z, ang, L):
        c = np.array([r_att * np.cos(np.radians(ang)), r_att * np.sin(np.radians(ang)), z])
        out = []
        while sum(len(o) for o in out) < N:
            p = rng.uniform(-1, 1, (4 * N, 3))
            p = p[np.sum(p ** 2, 1) <= 1] * L + c
            out.append(p[np.hypot(p[:, 0], p[:, 1]) > r_cyl])
        return np.vstack(out)[:N]
    d = cloud(0.0, 0.0, L_D)
    a = cloud(rise * dbp, twist * dbp + phi_deg, L_A)
    R = np.linalg.norm(d[rng.permutation(N)] - a, axis=1)
    return float(np.mean(1 / (1 + (R / R0) ** 6)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--bp", nargs="+", type=int, default=[5, 11, 20], help="base-pair separation, same order as --names")
    ap.add_argument("--names", nargs="+", default=["DNA5", "DNA11", "DNA20"])
    ap.add_argument("--out", default="results/DNA_validation")
    ap.add_argument("--S_lo", type=float, default=0.3); ap.add_argument("--S_hi", type=float, default=0.9)
    ap.add_argument("--R0", type=float, default=51.0); ap.add_argument("--L_D", type=float, default=20.0)
    ap.add_argument("--L_A", type=float, default=22.0)
    ap.add_argument("--tol", type=float, default=0.07, help="accepted |E_corr - E_AV| (E units)")
    ap.add_argument("--n_boot", type=int, default=300)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    folder = Path(a.folder); rng = np.random.default_rng(1)

    D = {}
    for nm in a.names:
        fr = next(p for p in folder.glob("*.csv") if p.name.upper().startswith(nm.upper() + "_APBS"))
        ft = next(p for p in folder.glob("*.csv") if p.name.upper().startswith(nm.upper() + "_DCBS"))
        D[nm] = dict(ref=load(fr), tgt=load(ft))
        al, de = leak_direct(*D[nm]["ref"])
        D[nm].update(alpha=al, delta=de)
    # alpha is a dye property -> pool; delta can differ per session -> pool as well (same session)
    al = np.mean([D[n]["alpha"] for n in a.names]); de = np.mean([D[n]["delta"] for n in a.names])

    def run(boot_idx=None):
        pts = []
        for n in a.names:
            DD, DA, AA = D[n]["tgt"]
            if boot_idx is not None:
                i = boot_idx[n]; DD, DA, AA = DD[i], DA[i], AA[i]
            pts.append(species_point(DD, DA, AA, al, de, a.S_lo, a.S_hi))
        return global_gamma(pts)
    g, b, a0, b0 = run()
    boots = []
    for _ in range(a.n_boot):
        idx = {n: rng.integers(0, len(D[n]["tgt"][0]), len(D[n]["tgt"][0])) for n in a.names}
        boots.append(run(idx)[:2])
    boots = np.array(boots); g_sd, b_sd = boots.std(0)

    rows = []
    for n, bp in zip(a.names, a.bp):
        DD, DA, AA = D[n]["tgt"]
        E, Eroll, f = corrected(DD, DA, AA, al, de, g, b, a.S_lo, a.S_hi)
        Eg = [corrected(DD, DA, AA, al, de, gg, b, a.S_lo, a.S_hi)[1] for gg in (g - g_sd, g + g_sd)]
        S_app = DA[f] / 1  # placeholder to keep arrays aligned
        nph = (DD + DA + AA)[f]
        Eapp = (DA[f] / (DA[f] + DD[f])).mean()
        hist, edges = np.histogram(E, bins=60, range=(-0.2, 1.2)); mode = 0.5 * (edges[:-1] + edges[1:])[hist.argmax()]
        R = a.R0 * (1 / Eroll - 1) ** (1 / 6)
        rows.append(dict(standard=n, bp=bp, bursts=int(f.sum()), E_apparent=Eapp, E_corrected=Eroll,
                         E_err_gamma=abs(Eg[0] - Eg[1]) / 2, E_median=float(np.median(E)), E_mode=float(mode),
                         sigma_obs=float(E.std()), sigma_shot=float(shot_sigma(E, nph)),
                         R_dye_dye_from_mean_E=R, axial_rise_A=3.4 * bp))
    T = pd.DataFrame(rows)

    # accessible-volume prediction, common phi
    phis = np.arange(0, 360, 10)
    rr = np.random.default_rng(7)
    grid = np.array([[av_E(bp, ph, a.L_D, a.L_A, a.R0, rr) for bp in T.bp] for ph in phis])
    rms = np.sqrt(((grid - T.E_corrected.to_numpy()) ** 2).mean(1))
    k = int(rms.argmin()); phi = float(phis[k])
    T["E_AV_best_phi"] = grid[k]; T["residual"] = T.E_corrected - T.E_AV_best_phi
    T["E_AV_phi_range"] = [f"{grid[:, j].min():.2f}-{grid[:, j].max():.2f}" for j in range(len(T))]
    # gamma scan: which gamma makes the ladder agree best with the AV model (best phi for each gamma)?
    gs = np.arange(0.4, 1.81, 0.05)
    def rms_at(gg):
        E = np.array([corrected(*D[n]["tgt"], al, de, gg, b, a.S_lo, a.S_hi)[1] for n in a.names])
        r = np.sqrt(((grid - E) ** 2).mean(1)); return r.min(), phis[r.argmin()]
    scan = np.array([rms_at(gg) for gg in gs])
    j = int(scan[:, 0].argmin()); g_av, rms_av, phi_av = gs[j], scan[j, 0], scan[j, 1]
    T["E_AV_best_phi"] = grid[k]; T["residual"] = T.E_corrected - T.E_AV_best_phi
    T["E_AV_phi_range"] = [f"{grid[:, jj].min():.2f}-{grid[:, jj].max():.2f}" for jj in range(len(T))]
    T["E_corrected_at_gamma_AV"] = [corrected(*D[n]["tgt"], al, de, g_av, b, a.S_lo, a.S_hi)[1] for n in a.names]
    # residue-residue distances (linker-averaged model of smfret_residue_distance.py), NOT the point-dye distance
    import importlib.util as _u
    _s = _u.spec_from_file_location("srd", Path(__file__).resolve().parent / "smfret_residue_distance.py")
    srd = _u.module_from_spec(_s); _s.loader.exec_module(srd)
    d_grid, E_grid = srd.calibration_curve(a.R0, srd.make_difference_vectors(a.L_D, a.L_A))
    def res_d(E):
        return srd.invert(float(E), d_grid, E_grid)[0] if 0 < E < 1 else np.nan
    T["residue_distance_A_corrected"] = [res_d(e) for e in T.E_corrected]
    T["residue_distance_A_apparent"] = [res_d(e) for e in T.E_apparent]
    T["residue_distance_A_at_gamma_AV"] = [res_d(e) for e in T.E_corrected_at_gamma_AV]
    mono = bool(np.all(np.diff(T.sort_values("bp").E_corrected.to_numpy()) < 0))
    ok_av = bool(np.abs(T.residual).max() <= a.tol)
    g_agree = bool(abs(g - g_av) <= max(0.2, 2 * g_sd))
    T.to_csv(out / "dna_validation.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].errorbar(T.bp, T.E_corrected, yerr=np.sqrt(T.E_err_gamma ** 2 + (T.sigma_obs / np.sqrt(T.bursts)) ** 2),
                   fmt="o", color="#1f77b4", label="corrected (global gamma)")
    ax[0].plot(T.bp, T.E_apparent, "s", color="grey", label="apparent")
    ax[0].plot(T.bp, T.E_AV_best_phi, "k^", label=f"AV model (phi={phi:.0f} deg)")
    ax[0].set_xlabel("base-pair separation"); ax[0].set_ylabel("E"); ax[0].legend(); ax[0].set_title("ladder vs AV model")
    ax[1].plot(gs, scan[:, 0], "-o", color="#1f77b4", ms=3); ax[1].axhline(a.tol, color="r", ls="--")
    ax[1].axvline(g, color="k", ls=":", label="regression gamma"); ax[1].legend()
    ax[1].set_xlabel("gamma"); ax[1].set_ylabel("rms E difference to AV model"); ax[1].set_title("which gamma fits the AV model")
    P = np.array([species_point(*D[n]["tgt"], al, de, a.S_lo, a.S_hi) for n in a.names])
    xs = np.linspace(P[:, 0].min() - .05, P[:, 0].max() + .05, 10)
    ax[2].plot(P[:, 0], P[:, 1], "o", color="#1f77b4"); ax[2].plot(xs, a0 + b0 * xs, "k-")
    ax[2].set_xlabel("P = DA'/(DA'+DD)"); ax[2].set_ylabel("AA/(DD+DA')")
    ax[2].set_title(f"global regression: gamma={g:.2f}+/-{g_sd:.2f}, beta={b:.2f}+/-{b_sd:.2f}")
    plt.tight_layout(); plt.savefig(out / "dna_validation.png", dpi=140)

    lines = [f"alpha = {al:.3f}  (per file: " + ", ".join(f"{D[n]['alpha']:.3f}" for n in a.names) + ")",
             f"delta = {de:.3f}  (per file: " + ", ".join(f"{D[n]['delta']:.3f}" for n in a.names) + ")",
             f"global gamma = {g:.2f} +/- {g_sd:.2f}, beta = {b:.2f} +/- {b_sd:.2f}", "",
             T.round(3).to_string(index=False), "",
             "RESIDUE-RESIDUE DISTANCES (A, linker-averaged model; point-dye value is R_dye_dye_from_mean_E above):",
             T[["standard", "bp", "axial_rise_A", "residue_distance_A_corrected", "residue_distance_A_apparent",
                "residue_distance_A_at_gamma_AV"]].round(1).to_string(index=False),
             "  corrected = regression gamma; apparent = no correction; at_gamma_AV = gamma that best fits the dye-position model", "",
             f"E decreases monotonically with bp: {mono}",
             f"AV best common phi = {phi:.0f} deg, max |residual| = {np.abs(T.residual).max():.3f} (tol {a.tol})",
             f"gamma that best matches the AV model = {g_av:.2f} (rms {rms_av:.3f}, phi {phi_av:.0f} deg); regression gamma = {g:.2f} +/- {g_sd:.2f}; agree: {g_agree}",
             f"widths vs shot noise: " + ", ".join(f"{r.standard} {r.sigma_obs:.3f} vs {r.sigma_shot:.3f}" for r in T.itertuples()),
             "", "VERDICT: " + ("PASS" if (mono and ok_av and g_agree) else "NOT VALIDATED") +
             " - ladder monotonic: %s; AV agreement at regression gamma: %s; regression gamma consistent with AV: %s" % (mono, ok_av, g_agree),
             "Caveat: AV model is crude (no steric path search, assumed attachment geometry and base-pair numbers); the DNA data are from one session (July 2023)."]
    (out / "dna_validation.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

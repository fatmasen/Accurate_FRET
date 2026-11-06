#!/usr/bin/env python3
"""
export_sigma.py - one table with the widths (sigma) of all distributions: DNA standards and proteins.

  python scripts/export_sigma.py                      (run from the project folder, after the protein analyses)
  python scripts/export_sigma.py --results results --dna "data/DNA standarts" --out results/sigma_export.csv

Rows in sigma_export.csv
  kind = DNA_standard      apparent-E width of each DNA standard (DCBS bursts, S 0.4-0.8, >= 200 photons: the selection of the
                           width table), plus the photon-shot-noise estimate -> this is what dna_sigma_floor.csv is made from
  kind = protein_merged    Gaussian states of the merged data of each results/<dataset> (E, sigma, fraction, sigma_floor used)
  kind = protein_measurement  each included measurement fitted on its own (BIC states) + its model-free standard deviation
sigma is in E units (corrected E for proteins, apparent E for DNA).
Only for your own analysis.
"""
import argparse, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cf", HERE / "correct_and_fit.py")
cf = importlib.util.module_from_spec(spec); spec.loader.exec_module(cf)


def fit_states(x, floor_table):
    cf.set_sigma_floor(floor_table, 0.005)
    w, mu, sd, _ = cf.select_model(x, 3, 0.10, 0.15, 10.0, verbose=False)
    return w, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--dna", default="data/DNA standarts")
    ap.add_argument("--out", default="results/sigma_export.csv"); ap.add_argument("--table", default="dna_sigma_floor.csv")
    ap.add_argument("--S_lo", type=float, default=0.4); ap.add_argument("--S_hi", type=float, default=0.8)
    ap.add_argument("--min_photons", type=int, default=200)
    a = ap.parse_args()
    rows = []

    # ---- DNA standards
    for f in sorted(Path(a.dna).glob("*DCBS*.csv")):
        d = pd.read_csv(f)
        d = d[(d.S_app > a.S_lo) & (d.S_app < a.S_hi) & (d.n_photons >= a.min_photons)]
        E = d.n_DA / (d.n_DA + d.n_DD)
        shot = float(np.sqrt(E.mean() * (1 - E.mean()) / np.median(d.n_DD + d.n_DA)))
        rows.append(dict(kind="DNA_standard", dataset="DNA standards", measurement=f.stem.replace("_DCBS", ""), state="single",
                         E=E.mean(), sigma=E.std(), fraction=1.0, sigma_shot_noise=shot, n_bursts=len(d),
                         note=f"apparent E, S {a.S_lo}-{a.S_hi}, >= {a.min_photons} photons"))

    # ---- proteins
    for d in sorted(p for p in Path(a.results).iterdir() if p.is_dir()):
        f = d / "merged_corrected_bursts.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        nph = float(np.median(t.n_DD + t.n_DA)); cf.set_shot_floor(nph)
        w, mu, sd = fit_states(t.E_corr.to_numpy(), a.table)
        for i in range(len(mu)):
            rows.append(dict(kind="protein_merged", dataset=d.name, measurement="all included", state=f"state{i + 1}", E=mu[i], sigma=sd[i],
                             fraction=w[i], sigma_floor=float(cf.sigma_floor(mu[i])[0]), sigma_shot_noise=float(np.sqrt(mu[i] * (1 - mu[i]) / nph)),
                             n_bursts=len(t), note="width basis: DNA table + shot noise"))
        for name, g in t.groupby("measurement"):
            x = g.E_corr.to_numpy()
            if len(x) < 300:
                continue
            w, mu, sd = fit_states(x, a.table)
            rows.append(dict(kind="protein_measurement", dataset=d.name, measurement=name, state="model-free", E=x.mean(), sigma=x.std(),
                             fraction=1.0, n_bursts=len(x), note="mean and standard deviation of all bursts of this measurement"))
            for i in range(len(mu)):
                rows.append(dict(kind="protein_measurement", dataset=d.name, measurement=name, state=f"state{i + 1}", E=mu[i], sigma=sd[i],
                                 fraction=w[i], sigma_floor=float(cf.sigma_floor(mu[i])[0]), n_bursts=len(x), note="BIC states"))
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(a.out, index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(out[out.kind != "protein_measurement"].round(3).to_string(index=False))
    print(f"\nSaved {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collects the results of all construct/state folders into results/FINAL_summary.csv (headline model-free E)
and results/FINAL_states.csv (descriptive Gaussian states, main fit and DNA-width sensitivity side by side)."""
from pathlib import Path
import pandas as pd
root = Path(__file__).resolve().parent.parent / "results"
S, T = [], []
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    mf = d / "model_free_summary.csv"
    if not mf.exists():
        continue
    r = pd.read_csv(mf).iloc[0].to_dict(); r["dataset"] = d.name
    ms = pd.read_csv(d / "measurement_summary.csv")
    r["n_measurements_total"] = len(ms); r["excluded"] = "; ".join(ms.loc[ms.merged == "NO", "measurement"])
    r["alpha_replaced"] = "; ".join(pd.read_csv(d / "alpha_delta_comparison.csv").query("alpha_replaced")["measurement"]) if (d / "alpha_delta_comparison.csv").exists() else ""
    S.append(r)
    cmp_ = d / "state_comparison_main_vs_alt_width.csv"
    dist = pd.read_csv(d / "residue_distances.csv") if (d / "residue_distances.csv").exists() else None
    if cmp_.exists():
        c = pd.read_csv(cmp_); c.insert(0, "dataset", d.name)
        if dist is not None and len(dist) == len(c):
            c["distance_A_main"] = dist["residue_residue_distance_A"].round(1).to_numpy()
        fl = d / "state_flags.txt"; c["flags"] = fl.read_text().replace("\n", " | ") if fl.exists() else ""
        T.append(c)
cols = ["dataset", "n_measurements", "n_measurements_total", "n_bursts", "E_mean_pooled", "E_median_pooled", "E_mode_pooled",
        "E_mean_sd_between_measurements", "E_mean_sem", "E_mean_shift_per_gamma_sd", "gamma", "gamma_sd", "excluded", "alpha_replaced"]
pd.DataFrame(S)[cols].round(3).to_csv(root / "FINAL_summary.csv", index=False)
if T:
    pd.concat(T).round(3).to_csv(root / "FINAL_states.csv", index=False)
print(pd.DataFrame(S)[cols[:9]].round(3).to_string(index=False))

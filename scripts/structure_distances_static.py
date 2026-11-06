#!/usr/bin/env python3
"""
structure_distances.py - CA-CA (and CB-CB) distances between the labelled residues of Trigger Factor in reported structures
(X-ray, NMR, cryo-EM).

  uv run scripts/structure_distances.py
  uv run scripts/structure_distances.py --ids 1w26 6d6s 7d80_bundle1 2mlx 2mly 2mlz --out results/structure_distances.csv

Residue pairs (from your construct names; the letter is checked against the structure, a mismatch means different numbering):
  E96-A160, A160-K327, T91-A230, E96-A230, A230-A297
Structures are taken from structures/<id>.cif|.pdb|.ent if present, otherwise downloaded once from files.rcsb.org (needs internet)
and saved into structures/. An id like 7d80_bundle1 first looks for a local file with exactly that name, then downloads 7d80.
NMR entries have many models: mean, SD, min and max over models are reported. Chains lacking the residues are skipped.
CA-CA is NOT the FRET distance; compare with the residue-residue distance (linker-averaged) from the FRET analysis.
Only for your own analysis.
"""
import argparse, re, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
from Bio.PDB import PDBParser, MMCIFParser

ONE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
       "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
DEFAULT_IDS = ["1w26", "6d6s", "7d80_bundle1", "2mlx", "2mly", "2mlz"]
DEFAULT_PAIRS = ["E96-A160", "A160-K327", "T91-A230", "E96-A230", "A230-A297"]


def find_or_download(sid, folder):
    for ext in (".cif", ".pdb", ".ent"):
        for stem in (sid, sid.lower(), sid.upper()):
            f = folder / (stem + ext)
            if f.exists():
                return f
    base = re.split(r"[_\\-]", sid)[0].lower()
    for ext in (".cif", ".pdb", ".ent"):
        f = folder / (base + ext)
        if f.exists():
            return f
    folder.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{base.upper()}.cif"
    dest = folder / (base + ".cif")
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def parse_pair(s):
    m = re.fullmatch(r"([A-Za-z])(\d+)-([A-Za-z])(\d+)", s)
    if not m:
        sys.exit(f"ERROR: pair '{s}' must look like E96-A160")
    return m.group(1).upper(), int(m.group(2)), m.group(3).upper(), int(m.group(4))


def atom(res, name):
    if name == "CB" and "CB" not in res:
        return res["CA"] if "CA" in res else None
    return res[name] if name in res else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", default=DEFAULT_IDS)
    ap.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    ap.add_argument("--dir", default="structures")
    ap.add_argument("--out", default="results/structure_distances.csv")
    ap.add_argument("--offset", type=int, default=0, help="add this number to the residue numbers (if a file is numbered differently)")
    ap.add_argument("--any_residue", action="store_true", help="also keep chains whose residue types do not match the construct names")
    a = ap.parse_args()
    rows = []
    for sid in a.ids:
        try:
            f = find_or_download(sid, Path(a.dir))
        except Exception as e:
            print(f"{sid}: could not get the structure ({e}). Download the file by hand into {a.dir}/")
            continue
        parser = MMCIFParser(QUIET=True) if f.suffix == ".cif" else PDBParser(QUIET=True)
        try:
            s = parser.get_structure(sid, str(f))
        except Exception as e:
            print(f"{sid}: cannot read {f.name} ({e})"); continue
        models = list(s)
        print(f"{sid}: {f.name}, {len(models)} model(s), chains " + ",".join(c.id for c in models[0]))
        for p in a.pairs:
            l1, n1, l2, n2 = parse_pair(p)
            for chain in models[0]:
                cid = chain.id
                ca, cb, names = [], [], None
                for m in models:
                    if cid not in m:
                        continue
                    ch = m[cid]
                    k1, k2 = (" ", n1 + a.offset, " "), (" ", n2 + a.offset, " ")
                    if k1 not in ch or k2 not in ch:
                        continue
                    r1, r2 = ch[k1], ch[k2]
                    names = (r1.get_resname(), r2.get_resname())
                    a1, a2 = atom(r1, "CA"), atom(r2, "CA")
                    if a1 is None or a2 is None:
                        continue
                    ca.append(a1 - a2)
                    b1, b2 = atom(r1, "CB"), atom(r2, "CB")
                    if b1 is not None and b2 is not None:
                        cb.append(b1 - b2)
                if not ca:
                    continue
                match = names == (ONE[l1], ONE[l2])
                if not match and not a.any_residue:
                    continue
                rows.append(dict(structure=sid, file=f.name, chain=cid, pair=p, residues_found=f"{names[0]}{n1 + a.offset}-{names[1]}{n2 + a.offset}",
                                 residue_types_match=match, n_models=len(ca), CA_CA_mean_A=np.mean(ca), CA_CA_sd_A=np.std(ca),
                                 CA_CA_min_A=np.min(ca), CA_CA_max_A=np.max(ca), CB_CB_mean_A=np.mean(cb) if cb else np.nan))
    out = pd.DataFrame(rows)
    if out.empty:
        print("\nNo pair found. Check the residue numbering (use --offset) or the structure files.")
        return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.round(2).to_csv(a.out, index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print("\n" + out.round(1)[["structure", "chain", "pair", "residues_found", "n_models", "CA_CA_mean_A", "CA_CA_sd_A", "CA_CA_min_A",
                                  "CA_CA_max_A", "CB_CB_mean_A"]].to_string(index=False))
    piv = out.groupby(["pair", "structure"]).CA_CA_mean_A.mean().unstack().round(1)
    print("\nCA-CA mean (A), averaged over chains:\n" + piv.to_string())
    piv.to_csv(Path(a.out).with_name(Path(a.out).stem + "_table.csv"))
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run the complete smFRET -> distance pipeline with one command.

  Step 1  scripts/correct_and_fit.py           corrections (alpha, delta, gamma, beta) + E histogram fit
  Step 2  scripts/smfret_residue_distance.py   E -> dye-dye and linker-corrected residue distances
  Step 3  scripts/smfret_vs_structure.py       comparison with a PDB structure (only if --pdb is given)

ONE FOLDER PER STATE (recommended for many data sets)
  data/
  ├── apo/          apo_dcbs_1.csv  apo_dcbs_2.csv  apo_apbs.csv
  ├── client_A/     client_dcbs_1.csv ... client_apbs.csv
  └── ...
  Every CSV in a state folder is pooled as ONE state. Files whose name contains "apbs" are used as the
  reference (donor-only / acceptor-only / FRET bursts); all other CSVs are analysed. Results go to
  results/<folder name>/ and, for several folders, a combined table summary_all_states.csv.

  python run_analysis.py --folder apo                      one state
  python run_analysis.py --folder apo client_A             several states
  python run_analysis.py --all                             every sub-folder of the data folder
  python run_analysis.py --make_folders apo client_A       just create the (empty) state folders

  Folder options:  --ref_pattern "*apbs*" (default)  --target_pattern "*dcbs*" (default: all other CSVs)
                   --ref shared_apbs.csv  (use one reference for all folders)  --self_ref (targets contain
                   the donor-only/acceptor-only bursts themselves)

SINGLE FILES (as before)
  python run_analysis.py --target APO_AK_DCBS.csv --ref APO_AK_APBS.csv --name apo
  python run_analysis.py --target rep1.csv rep2.csv rep3.csv --ref APBS.csv --name apo      (several files = ONE state)
  python run_analysis.py --target "rep*.csv" --ref APBS1.csv APBS2.csv --name apo          (wildcards)

Common options (apply to every analysis): --n auto|1|2|3  --names low high  --R0 51  --L_D 20  --L_A 22
  --pdb 1w26.pdb --site1 A:150 --site2 A:380   (structure comparison)
Options not listed here are passed on to scripts/correct_and_fit.py, for example:
  --max_n 2 --min_frac 0.05 --min_sep 0.15 --delta_bic 10   number-of-states selection
  --estimator mode|median|mean|gauss --kde_bw 0.02          how E is reported for a single state
  --gamma 1.1 --gamma_err 0.05 --S_min 0.3 --S_max 0.7 --bg_DD --bg_DA --bg_AA --n_boot
"""
import argparse
import fnmatch
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS, DATA, STRUCT, RESULTS = (ROOT / "scripts", ROOT / "data",
                                  ROOT / "structures", ROOT / "results")


def find(name, folder):
    p = Path(name)
    if p.exists():
        return p.resolve()
    q = folder / name
    if q.exists():
        return q.resolve()
    sys.exit(f"ERROR: file '{name}' not found (looked in the current folder and in {folder})")


def expand(specs, folder, label):
    """File names or wildcard patterns -> list of existing files (looked up in the given folder or as typed)."""
    out = []
    for sp in specs:
        if any(c in sp for c in "*?["):
            hits = sorted(Path(".").glob(sp)) + sorted(folder.glob(sp))
            hits = sorted({h.resolve() for h in hits})
            if not hits:
                sys.exit(f"ERROR: no {label} file matches '{sp}' (looked in the current folder and in {folder})")
            out += hits
        else:
            out.append(find(sp, folder))
    return out


class StepFailed(Exception):
    pass


def run(cmd, cwd, log):
    cmd = [str(c) for c in cmd]
    header = "\n$ " + " ".join(cmd) + "\n"
    print(header)
    log.write(header)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", MPLBACKEND="Agg")
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    out = p.stdout + p.stderr
    print(out.encode(sys.stdout.encoding or "utf-8", "replace").decode(sys.stdout.encoding or "utf-8"))
    log.write(out)
    log.flush()
    if p.returncode != 0:
        raise StepFailed(f"a step failed (see {Path(cwd) / 'run_log.txt'})")


# ------------------------------------------------------------------ state folders
def list_state_folders():
    return sorted(p for p in DATA.iterdir()
                  if p.is_dir() and not p.name.startswith((".", "_")) and any(p.glob("*.csv")))


def resolve_folder(name):
    p = Path(name)
    if p.is_dir():
        return p.resolve()
    q = DATA / name
    if q.is_dir():
        return q.resolve()
    avail = ", ".join(f.name for f in list_state_folders()) or "none yet"
    sys.exit(f"ERROR: folder '{name}' not found (looked in the current folder and in {DATA}).\n"
             f"       State folders in the data folder: {avail}\n"
             f"       Create one with:  python run_analysis.py --make_folders {name}")


def csv_files(folder):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv")


def folder_inputs(folder, args):
    files = csv_files(folder)
    own_refs = [p for p in files if fnmatch.fnmatch(p.name.lower(), args.ref_pattern.lower())]
    if args.target_pattern:
        targets = [p for p in files if fnmatch.fnmatch(p.name.lower(), args.target_pattern.lower())]
    else:
        targets = [p for p in files if p not in own_refs]
    if args.ref:                                         # explicit shared reference wins
        refs = expand(args.ref, DATA, "reference")
    else:
        refs = own_refs
    if not targets:
        raise ValueError(f"no data CSV in {folder} (reference pattern '{args.ref_pattern}', "
                         f"target pattern '{args.target_pattern or 'all other CSVs'}')")
    if not refs and not args.self_ref:
        raise ValueError(f"no reference file in {folder}: put an APBS file there (name containing "
                         f"'{args.ref_pattern.strip('*')}'), pass --ref <file>, or use --self_ref if the data "
                         f"files themselves contain donor-only and acceptor-only bursts")
    return targets, refs


# ------------------------------------------------------------------ per-measurement analysis
def analyze_per_measurement(name, folder, args, extra):
    """Every dcbs/apbs pair is corrected with its own factors, compatible ones are merged (per_measurement.py)."""
    outdir = RESULTS / name
    outdir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    with open(outdir / "run_log.txt", "a", encoding="utf-8") as log:
        log.write(f"\n{'=' * 70}\nRun {datetime.now():%Y-%m-%d %H:%M:%S} (per measurement)\nfolder={folder}\n")
        cmd = [py, SCRIPTS / "per_measurement.py", "--folder", folder, "--out", outdir, "--n", str(args.n),
               "--R0", args.R0, "--R0_err", args.R0_err, "--L_D", args.L_D, "--L_A", args.L_A]
        if args.names:
            cmd += ["--names", *args.names]
        if args.gamma is not None:
            cmd += ["--gamma", args.gamma, "--gamma_err", args.gamma_err]
        run(cmd + extra, outdir, log)
        if args.pdb:
            if not (args.site1 and args.site2):
                raise ValueError("--pdb needs --site1 and --site2 (e.g. --site1 A:150 --site2 A:380)")
            pdb = find(args.pdb, STRUCT)
            run([py, SCRIPTS / "smfret_vs_structure.py", "data.csv", "--pdb", pdb,
                 "--site1", args.site1, "--site2", args.site2, "--R0", args.R0,
                 "--linker", args.linker, "--av", "--calibrate",
                 "--out", "structure_comparison"], outdir, log)
    return outdir


# ------------------------------------------------------------------ one analysis
def analyze(name, targets, refs, args, extra):
    outdir = RESULTS / name
    outdir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    with open(outdir / "run_log.txt", "a", encoding="utf-8") as log:
        log.write(f"\n{'=' * 70}\nRun {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                  f"targets={[str(t) for t in targets]}\nrefs={[str(x) for x in refs]}\n")
        cmd = [py, SCRIPTS / "correct_and_fit.py", *targets, "--n", str(args.n), "--out", "data.csv"]
        if refs:
            cmd += ["--ref", *refs]
        if args.names:
            cmd += ["--names", *args.names]
        if args.centers:
            cmd += ["--centers", *args.centers]
        if args.gamma is not None:
            cmd += ["--gamma", args.gamma, "--gamma_err", args.gamma_err]
        cmd += extra
        run(cmd, outdir, log)

        run([py, SCRIPTS / "smfret_residue_distance.py", "data.csv", "--R0", args.R0,
             "--R0_err", args.R0_err, "--L_D", args.L_D, "--L_A", args.L_A,
             "--out", "residue_distances"], outdir, log)

        if args.pdb:
            if not (args.site1 and args.site2):
                raise ValueError("--pdb needs --site1 and --site2 (e.g. --site1 A:150 --site2 A:380)")
            pdb = find(args.pdb, STRUCT)
            run([py, SCRIPTS / "smfret_vs_structure.py", "data.csv", "--pdb", pdb,
                 "--site1", args.site1, "--site2", args.site2, "--R0", args.R0,
                 "--linker", args.linker, "--av", "--calibrate",
                 "--out", "structure_comparison"], outdir, log)
    return outdir


def write_summary(done):
    """Combine the residue_distances.csv of all analysed state folders into one table."""
    import pandas as pd
    frames = []
    for name, outdir in done:
        f = outdir / "residue_distances.csv"
        if f.exists():
            d = pd.read_csv(f)
            d.insert(0, "state_folder", name)
            frames.append(d)
    if not frames:
        return
    s = pd.concat(frames, ignore_index=True)
    keep = [c for c in ("state_folder", "state", "E", "E_err", "population_pct", "population_err_pct",
                        "dye_dye_R_A", "residue_residue_distance_A", "distance_min_A", "distance_max_A", "note")
            if c in s.columns]
    s = s[keep]
    s.to_csv(RESULTS / "summary_all_states.csv", index=False)
    show = s.copy()
    for c in ("E", "E_err"):
        show[c] = show[c].round(3)
    for c in show.columns:
        if c not in ("state_folder", "state", "E", "E_err", "note"):
            show[c] = show[c].round(1)
    print("\n" + "=" * 100 + "\nSUMMARY OF ALL STATES\n" + "=" * 100)
    print(show.drop(columns=[c for c in ("note",) if c in show.columns]).to_string(index=False))
    print(f"\nSaved {RESULTS / 'summary_all_states.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", nargs="+", help="state folder(s) inside the data folder; all CSVs in a folder = ONE state")
    ap.add_argument("--per_measurement", action="store_true",
                    help="analyse every apbs/dcbs pair of a folder separately (own alpha/delta/gamma/beta), "
                         "exclude incompatible ones and merge the rest (scripts/per_measurement.py). "
                         "Extra options: --exclude NAME.. --include NAME.. --w1_max 0.07 --gamma_tol 0.15")
    ap.add_argument("--all", action="store_true", help="analyse every sub-folder of the data folder (skips names starting with _ or .)")
    ap.add_argument("--make_folders", nargs="+", metavar="NAME", help="create empty state folder(s) in the data folder and stop")
    ap.add_argument("--ref_pattern", default="*apbs*", help="file-name pattern of the reference files inside a state folder")
    ap.add_argument("--target_pattern", help="file-name pattern of the data files (default: all CSVs that are not references)")
    ap.add_argument("--self_ref", action="store_true", help="no separate reference: the data files contain donor-only/acceptor-only bursts")
    ap.add_argument("--target", nargs="+", help="burst CSV file(s) to analyse; several files = ONE state (pooled); wildcards allowed")
    ap.add_argument("--ref", nargs="+", help="APBS CSV file(s) with donor-only/acceptor-only/FRET bursts; wildcards allowed")
    ap.add_argument("--name", help="results sub-folder name (default: folder name / target file name, 'combined' for several files)")
    ap.add_argument("--n", default="auto", help="'auto' (default: 1, 2 or 3 states chosen by BIC) or a fixed number of states")
    ap.add_argument("--names", nargs="+", help="state names, low E -> high E")
    ap.add_argument("--centers", nargs="+", type=float, help="starting E values for the fit")
    ap.add_argument("--gamma", type=float, help="fix gamma (default: estimate from reference)")
    ap.add_argument("--gamma_err", type=float, default=0.0)
    ap.add_argument("--R0", type=float, default=51.0, help="Forster radius (A), Alexa555/647")
    ap.add_argument("--R0_err", type=float, default=2.0)
    ap.add_argument("--L_D", type=float, default=20.0, help="donor linker length (A)")
    ap.add_argument("--L_A", type=float, default=22.0, help="acceptor linker length (A)")
    ap.add_argument("--linker", type=float, default=20.0, help="linker length for the structure AV model (A)")
    ap.add_argument("--pdb", help="PDB file (in structures/) for step 3")
    ap.add_argument("--site1", help="labeled residue 1, CHAIN:RESNUM (e.g. A:150)")
    ap.add_argument("--site2", help="labeled residue 2, CHAIN:RESNUM (e.g. A:380)")
    args, extra = ap.parse_known_args()

    # ---- just create folders
    if args.make_folders:
        for nm in args.make_folders:
            (DATA / nm).mkdir(parents=True, exist_ok=True)
            print(f"created {DATA / nm}")
        print("\nPut the CSV files of each state into its folder (reference file name containing 'apbs'), then run e.g.\n"
              f"  python run_analysis.py --folder {args.make_folders[0]}\n  python run_analysis.py --all")
        return

    DATA.mkdir(parents=True, exist_ok=True)
    jobs = []        # (name, targets, refs)
    if args.folder or args.all:
        folders = [resolve_folder(f) for f in args.folder] if args.folder else list_state_folders()
        if args.all and not folders:
            sys.exit(f"ERROR: no sub-folders with CSV files in {DATA}")
        if args.name and len(folders) > 1:
            sys.exit("ERROR: --name can only be used with a single folder (the folder name is used otherwise)")
        for f in folders:
            jobs.append((args.name or f.name, f, None))
    elif args.target:
        targets = expand(args.target, DATA, "target")
        refs = expand(args.ref, DATA, "reference") if args.ref else []
        name = args.name or (targets[0].stem if len(targets) == 1 else "combined")
        jobs.append((name, (targets, refs), None))
    else:
        ap.error("give --folder <name> (or --all), or --target <file(s)>")

    done, failed = [], []
    for name, spec, _ in jobs:
        try:
            if isinstance(spec, Path) and args.per_measurement:
                print("\n" + "#" * 100 + f"\n# STATE FOLDER '{name}': per-measurement analysis\n" + "#" * 100)
                outdir = analyze_per_measurement(name, spec, args, extra)
                done.append((name, outdir))
                print(f"\nDone '{name}'. Results are in: {outdir}")
                continue
            if isinstance(spec, Path):
                targets, refs = folder_inputs(spec, args)
                print("\n" + "#" * 100 + f"\n# STATE FOLDER '{name}': {len(targets)} data file(s), "
                      f"{len(refs)} reference file(s)\n" + "#" * 100)
                for t in targets:
                    print(f"   data:      {t.name}")
                for r in refs:
                    print(f"   reference: {r.name}")
            else:
                targets, refs = spec
            outdir = analyze(name, targets, refs, args, extra)
            done.append((name, outdir))
            print(f"\nDone '{name}'. Results are in: {outdir}")
        except (ValueError, StepFailed) as e:
            failed.append((name, str(e)))
            print(f"\nERROR in '{name}': {e}")

    if len(jobs) > 1 and done:
        write_summary(done)
    if failed:
        print("\nFAILED:")
        for n_, msg in failed:
            print(f"   {n_}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()

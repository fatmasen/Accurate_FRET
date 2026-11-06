# Accurate smFRET analysis (Alexa555 / Alexa647)

Converts burst-wise smFRET data (ALEX/PIE, exported as APBS and DCBS burst tables) into corrected ("accurate")
FRET efficiencies, E histograms with fitted states, and residue-residue distances. Optionally compares the
distances with PDB structures.

The correction follows Lee et al. 2005 (Biophys. J. 88, 2939): leakage (alpha), direct excitation (delta) and the
detection/quantum-yield factors (gamma, beta) are taken from the APBS reference bursts of the same measurement.

## Folder layout

```
Accurate_FRET/
├── run_analysis.py             runner (one measurement or one folder of measurements)
├── requirements.txt            Python packages
├── run_all.sh                  runs every data folder (bash; on Windows see "Run all folders")
├── data/                       your burst CSV files (not part of the repository)
│   ├── dna_sigma_floor.csv     widths of the static DNA standards (E,sigma), used as the width floor
│   ├── DNA standarts/          DNA5/11/20 _APBS.csv and _DCBS.csv
│   └── <construct>/<state>/    e.g. TF_E96A160/Monomer: pairs  <name>_apbs.csv + <name>_dcbs.csv
├── structures/                 PDB/mmCIF files (not part of the repository)
├── scripts/
│   ├── correct_and_fit.py            corrections + Gaussian-mixture fit (core)
│   ├── per_measurement.py            every measurement analysed alone, compatible ones merged
│   ├── smfret_residue_distance.py    E -> dye-dye and residue-residue distance
│   ├── smfret_vs_structure.py        comparison with one PDB structure
│   ├── structure_distances.py        CA-CA / CB-CB distances of labelled residues in many structures
│   ├── validate_dna.py               check of the correction chain on the DNA standards
│   ├── export_sigma.py               table of all widths (DNA standards and proteins)
│   ├── collect_results.py            collects all result folders into FINAL_summary.csv / FINAL_states.csv
│   └── ptu_dye_factors.py            optional: alpha/delta from PicoQuant .ptu files (needs ptufile)
└── results/                    created automatically, one sub-folder per analysis
```

## Setup (once)

```
uv pip install -r requirements.txt
```
(or `python -m pip install -r requirements.txt`). `ptufile` is only needed for `ptu_dye_factors.py`.

## Run

From the project folder.

**All measurements of one folder (recommended):**
```
uv run run_analysis.py --folder TF_E96A160/Monomer --per_measurement
```
Each measurement is analysed alone, compared with the others, compatible ones are merged, and the merged data are
fitted. File pairs must have names that differ only by `apbs` / `dcbs`; an unpaired file stops the run.
Without the runner, with your own output folder:
```
uv run scripts/per_measurement.py --folder TF_E96A160/Monomer --out results/TF_E96A160_Monomer
```

**One measurement:**
```
uv run run_analysis.py --target <name>_DCBS.csv --ref <name>_APBS.csv --name mymeasurement
```
With a structure comparison add `--pdb 1w26.pdb --site1 A:96 --site2 A:160`.

**Run all folders (PowerShell):**
```
Get-ChildItem data\TF_* -Directory | ForEach-Object { $c = $_.Name; Get-ChildItem $_.FullName -Directory | ForEach-Object { uv run scripts/per_measurement.py --folder "$c/$($_.Name)" --out "results/${c}_$($_.Name)" } }
uv run scripts/collect_results.py
```

**DNA standards check and width table:**
```
uv run scripts/validate_dna.py --folder "data/DNA standarts" --out results/DNA_validation
uv run scripts/export_sigma.py
```

**Distances in reported structures:**
```
uv run scripts/structure_distances.py --ids 1w26 6d6s 2mlx
```

## Definitions used

| Quantity | Definition |
|---|---|
| alpha | per file, from its own APBS donor-only bursts (S > 0.9). Outlier check against files of the same laser setting (highS / normal, needs at least 4 files): beyond max(3 MAD, 0.02) the group median is used and the file is flagged |
| delta | per file, from its own APBS acceptor-only bursts (S < 0.2); never replaced. The log compares it with the group median and with the DNA controls (information only) |
| gamma | per file, from its own APBS regression of AA/(DD+DA') against DA'/(DA'+DD) (`--gamma_mode per_file`, default). If that regression is unreliable the folder median is used; a file more than 15 % from the folder median is flagged. `--gamma_mode global` uses one value from all folders, `--gamma` a value you measured |
| beta | per file, from its own APBS given gamma |
| E, S | DA' = DA - alpha DD - delta AA; E = DA' / (DA' + gamma DD); S = 1 / (1 + AA / (beta (DA' + gamma DD))). Distances only for 0.15 < E < 0.85 |
| width (sigma) of a state | free, but not below the static DNA standards (`data/dna_sigma_floor.csv`, scaled with sqrt(E(1-E)) outside its range) and not below photon shot noise. `--width_basis shot` uses shot noise only |
| number of states | automatic (Gaussian mixture, BIC with guards: `--min_frac 0.10`, `--min_sep 0.15`, `--delta_bic 10`); override with `--n` or `--max_n` |
| compatibility of measurements | Wasserstein distance of the E distributions (`--w1_max 0.07`; `--w1_max 1` merges everything), at least `--min_bursts 300` bursts |
| distance | R = R0 (1/E - 1)^(1/6) for the dye-dye distance (R0 = 51 A); residue-residue distance from a dye-linker model (spheres of 20 A donor, 22 A acceptor), `--R0`, `--L_D`, `--L_A` |

## Output files (in `results/<name>/`)

| File | Content |
|---|---|
| `measurement_summary.csv` | per measurement: correction factors, bursts, states, compatibility, merged yes/no |
| `alpha_delta_comparison.csv` | alpha and delta of each file against the group median and the DNA controls |
| `model_free_summary.csv`, `per_measurement_E.csv` | pooled mean, median and mode of E, spread between measurements (headline result) |
| `data.csv` | Gaussian states: E, error, width, width floor, fraction, error |
| `residue_distances.csv` / `.png` | dye-dye and residue-residue distance per state, with range |
| `merged_corrected_bursts.csv` | all merged bursts with corrected E and S |
| `merged_FRET.png`, `per_measurement_histograms.png`, `factors_per_measurement.png`, `uncorrected_vs_corrected.png`, `apbs_uncorrected.png` | E-S plot, histograms and fits, factors per measurement, correction effect |
| `state_comparison_main_vs_alt_width.csv`, `state_flags.txt`, `alt_width_sensitivity/` | same bursts refitted with the other width basis, and flags for unstable states |
| `run_log.txt` | full log (commands, factors, comments, results) |

## How to read the results

- **The headline numbers are the model-free ones:** pooled mean / median / mode of E and the spread between
  measurements. The Gaussian states are a description of the histogram.
- **State positions and populations depend on the width assumption.** The sensitivity fit and `state_flags.txt`
  show how much each state moves; a state that moves by more than about 0.05 in E is not robust.
- **A state at the width floor** is at least as narrow as a static molecule; a state much broader than the floor
  contains dynamics or several conformations.
- **Error bars on distances** combine the error of the mean E and the R0 uncertainty (+/- 2 A). They are not the
  spread of distances within a state.

## Input file format

CSV with the columns `n_DD, n_DA, n_AA, E_app, S_app, n_photons, time_length, filename` (comma/dot and
semicolon/decimal-comma formats are both read).

## Known limitations

- Correction factors from the APBS reference only hold for data taken with the same lasers, filters, detectors and
  alignment. delta follows the laser power ratio; alpha and gamma should not.
- gamma from a single file can be poorly determined, especially when the file contains a narrow range of E.
  The DNA standards in this repository come from a different session than the protein data and do **not**
  validate gamma (the regression and a dye-position model disagree); `validate_dna.py` reports this.
  Absolute E and distances are therefore uncertain; differences between constructs and conditions are more robust.
- The dye-linker model is a free-space approximation and biases absolute distances by several A, more at E > 0.85
  and E < 0.15. For publication consider accessible-volume modelling (FPS, FRETraj).
- A broad E distribution can be several states or dynamics. Check lifetimes, BVA or PDA before interpreting
  states as separate conformations.
- Numbering in a PDB file can differ from the construct numbering; `structure_distances.py` checks the residue types.

## Citation

If you use this analysis, cite Lee et al. 2005 (correction scheme) and state the version (git commit) of this
repository. Add your own reference here.

#!/bin/bash
# runs the final analysis for every construct/state folder (global gamma is scanned once and cached in results/global_gamma.csv)
cd "$(dirname "$0")"
for d in data/TF_*/*/; do
  n=$(echo "$d" | sed 's#data/##;s#/$##;s#/#_#')
  python3 -I scripts/per_measurement.py --folder "$d" --out "results/$n" > "results/$n.log" 2>&1
  echo "$n done"
done

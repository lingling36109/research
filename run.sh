#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH --gpus=1
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -o slurm-%j.out

set -euo pipefail

module load cudatoolkit gcc

MATRICES=(144 road_usa cant belgium_osm)
MAT_DIR="${PSCRATCH}/matrices"
mkdir -p results

for m in "${MATRICES[@]}"; do
    cfg="results/${m}_config.json"
    if command -v jq >/dev/null 2>&1; then
        jq --arg f "results/${m}_stats.json" \
           '.stats_output_file=$f' config/bench_seg.json > "$cfg"
    else
        sed "s|\"stats_output_file\": \"stats.json\"|\"stats_output_file\": \"results/${m}_stats.json\"|" \
            config/bench_seg.json > "$cfg"
    fi
    echo "=== ${m} ==="
    srun ./spgemm "${MAT_DIR}/${m}.csr" "${MAT_DIR}/${m}.csr" "$cfg" \
        | tee "results/${m}_run.log"
done

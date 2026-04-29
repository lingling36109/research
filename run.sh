#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH --gpus=1
#SBATCH -t 00:15:00
module load cudatoolkit gcc
srun ./spgemm $PSCRATCH/matrices/A.csr --config config/analysis.json
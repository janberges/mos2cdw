#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node 96
#SBATCH --partition standard96:test
#SBATCH --time 01:00:00

module load anaconda3 intel impi

cd $SLURM_SUBMIT_DIR

export SLURM_CPU_BIND=none

mpirun python3 phases.py

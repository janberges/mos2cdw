#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node 96
#SBATCH --partition standard96:test
#SBATCH --time 01:00:00

module load intel-oneapi-compilers
module load intel-oneapi-mpi
module load intel-oneapi-mkl
module load miniconda3

cd $SLURM_SUBMIT_DIR

export SLURM_CPU_BIND=none
export MKL_NUM_THREADS=192
export NUMEXPR_NUM_THREADS=192
export OMP_NUM_THREADS=192
export OPENBLAS_NUM_THREADS=192
export VECLIB_MAXIMUM_THREADS=192

python3 dosef.py

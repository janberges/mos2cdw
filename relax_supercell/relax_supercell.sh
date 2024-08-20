#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node 96
#SBATCH --partition standard96
#SBATCH --time 12:00:00

module load intel-oneapi-compilers
module load intel-oneapi-mpi
module load intel-oneapi-mkl
module load miniconda3

cd $SLURM_SUBMIT_DIR

export SLURM_CPU_BIND=none
export MKL_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24
export OMP_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export VECLIB_MAXIMUM_THREADS=24

python3 relax_12sqrt3.py > relax_12sqrt3.out &
python3 relax_18sqrt3.py > relax_18sqrt3.out &
python3 relax_12.py > relax_12.out &
python3 relax_18.py > relax_18.out &

wait

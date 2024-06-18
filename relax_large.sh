#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node 96
#SBATCH --partition standard96
#SBATCH --time 12:00:00

module load anaconda3 intel impi

cd $SLURM_SUBMIT_DIR

export SLURM_CPU_BIND=none
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export VECLIB_MAXIMUM_THREADS=16

for i in `seq 1 6`
do
    echo "Starting calculation for doping $i.."
    python3 relax_large.py $i > relax_large_$i.out &
done

wait

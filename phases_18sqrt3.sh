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
export MKL_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export VECLIB_MAXIMUM_THREADS=6

#ini=2x2
ini=triangle
#nel=`seq -w 0 6 186`
#nel="`seq 192 6 246` `seq 348 6 474`"
nel=`seq 247 278`
#nel=`seq 279 310`
#nel=`seq 311 342`
#nel=`seq 480 6 666`

for n in $nel
do
    python3 18sqrt3.py $n $ini > phases_18sqrt3/dop_${n}_from_${ini}.out &
done

wait

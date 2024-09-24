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

#for nel in `seq -w 0 6 186`
#for nel in `seq 192 6 240` `seq 342 6 474`
#for nel in `seq 288 6 474`
#for nel in `seq 480 6 666`
for nel in `seq 243 274`
#for nel in `seq 275 306`
#for nel in `seq 307 338`
do
    python3 18sqrt3.py $nel > 18sqrt3_$nel.out &
done

wait

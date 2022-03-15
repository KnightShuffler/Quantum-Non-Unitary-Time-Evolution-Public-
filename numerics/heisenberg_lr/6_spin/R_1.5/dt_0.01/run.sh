#!/bin/bash
#SBATCH --qos=parallel
#SBATCH --partition=parallel
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH --mem=256000
#SBATCH --reservation=mmotta_109

export MKL_NUM_THREADS=28
srun python test.py > test.out

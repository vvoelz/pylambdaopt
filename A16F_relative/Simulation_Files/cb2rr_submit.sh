#!/bin/bash

#SBATCH --job-name=EE
#SBATCH --output=EE.out
#SBATCH --partition=normal
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00

cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=1

module load gromacs
module load mpi/openmpi
srun gmx_mpi mdrun -v -deffnm EE -dhdl EE_2dhdl.xvg -ntomp $OMP_NUM_THREADS


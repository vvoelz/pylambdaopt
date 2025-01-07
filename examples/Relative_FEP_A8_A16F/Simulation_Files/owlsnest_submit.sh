#!/bin/bash
#PBS -N EE
#PBS -o EE.out
#PBS -e EE.err
#PBS -q normal
#PBS -l nodes=4:ppn=28
#PBS -l walltime=48:00:00

cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=1

module load gromacs
module load mpi/openmpi
mpirun -np $(($PBS_NP)) gmx_mpi mdrun -v -deffnm EE_opt -dhdl EE_opt_5dhdl.xvg -ntomp $OMP_NUM_THREADS


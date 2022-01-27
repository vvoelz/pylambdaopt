#!/bin/bash

#PBS -N round5_opt_RUN5.pf
#PBS -o round5_opt_RUN5.out
#PBS -q normal
#PBS -l nodes=1:ppn=28
#PBS -l walltime=48:00:00
#PBS

cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=1

module load gromacs/2020.3
module load mpi/openmpi

##gmx mdrun -v -deffnm npt
##mpirun mdrun_mpi -maxh 47 -s prod.tpr
mpirun mdrun_mpi -maxh 47 -s prod.tpr -cpi state.cpt

#~/packages/anaconda3/bin/python opt.py
##~/packages/anaconda3/envs/openff/bin/python example_analysis.py

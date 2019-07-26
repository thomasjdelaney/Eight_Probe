#!/bin/bash
$PBS_O_WORKDIR=$HOME/Eight_Probe

# request resources:
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00
# on compute node, change directory to 'submission directory':
cd $PBS_O_WORKDIR
# run your program, timing it for good measure:
time $HOME/Eight_Probe/py/bin_width_variation.py

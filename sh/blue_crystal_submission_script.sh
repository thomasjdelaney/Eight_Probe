#!/bin/bash
PBS_O_WORKDIR=Eight_Probe

# request resources:
#PBS -l nodes=4:ppn=16
#PBS -l walltime=04:00:00
# on compute node, change directory to 'submission directory':
cd $PBS_O_WORKDIR
# run your program, timing it for good measure:
time python -m scoop $HOME/Eight_Probe/py/bin_width_variation.py

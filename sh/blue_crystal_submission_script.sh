#!/bin/bash
#
# request resources:
#PBS -l nodes=4:ppn=16,walltime=01:00:00

# on compute node, change directory to 'submission directory':

export nodes=`cat $PBS_NODEFILE`
export nnodes=`cat $PBS_NODEFILE | wc -l`
export confile=inf.$PBS_JOBID.conf

for i in $nodes
do
  echo ${i}>>$confile
done

# run your program, timing it for good measure:
time python -m scoop --hostfile $confile -n $nnodes $HOME/Eight_Probe/py/bin_width_variation.py -n 0

#!/bin/bash
#
# request resources:
#PBS -l nodes=1:ppn=16,walltime=04:00:00

# on compute node, change directory to 'submission directory':

export nodes=`cat $PBS_NODEFILE`
export nnodes=`cat $PBS_NODEFILE | wc -l`
export confile=inf.$PBS_JOBID.conf

for i in $nodes
do
  echo ${i}>>$confile
done

# run your program, timing it for good measure:
# NB: Only using 50 cells here because of memory constraints
time python3 $HOME/Eight_Probe/py/plot_communities.py -r total -c rectified
echo `date +'%d-%m-%Y %T'`" INFO: Completed total rectified."
time python3 $HOME/Eight_Probe/py/plot_communities.py -r conditional -c rectified
echo `date +'%d-%m-%Y %T'`" INFO: Completed conditional rectified."
time python3 $HOME/Eight_Probe/py/plot_communities.py -r signal -c rectified 
echo `date +'%d-%m-%Y %T'`" INFO: Completed signal rectified."
time python3 $HOME/Eight_Probe/py/plot_communities.py -r total -c negative
echo `date +'%d-%m-%Y %T'`" INFO: Completed total negative."
time python3 $HOME/Eight_Probe/py/plot_communities.py -r conditional -c negative
echo `date +'%d-%m-%Y %T'`" INFO: Completed conditional negtive."
time python3 $HOME/Eight_Probe/py/plot_communities.py -r signal -c negative
echo `date +'%d-%m-%Y %T'`" INFO: Completed signal negative."
time python3 $HOME/Eight_Probe/py/plot_communities.py -r total -c absolute 
echo `date +'%d-%m-%Y %T'`" INFO: Completed total absolute."
time python3 $HOME/Eight_Probe/py/plot_communities.py -r conditional -c absolute 
echo `date +'%d-%m-%Y %T'`" INFO: Completed conditional absolute."
time python3 $HOME/Eight_Probe/py/plot_communities.py -r signal -c absolute 
echo `date +'%d-%m-%Y %T'`" INFO: Completed signal absolute."
echo `date +'%d-%m-%Y %T'`" INFO: Done."




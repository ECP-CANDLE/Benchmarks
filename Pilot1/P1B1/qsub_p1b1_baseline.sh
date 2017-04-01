#!/bin/bash -l
#COBALT -A nci_doe_pilot1
#COBALT -n 1 
#COBALT -t 00:30:00
#COBALT -M brettin@anl.gov
#COBALT --jobname Candle-ECP-P1B1 
#COBALT -q debug

echo $PATH
echo "activiting keras2"
# source activate keras2

THIS=$( cd $( dirname $0 ); /bin/pwd )
PYTHONPATH=$PYTHONPATH
PYTHONPATH+=:$THIS:
PYTHONPATH+=:$THIS/../lib
export PYTHONPATH=$PYTHONPATH

echo "$PYTHONPATH"
which python

python /projects/Candle_ECP/brettin/benchmarks/Pilot1/P1B1/p1b1_baseline_keras2.py 

# source deactivate

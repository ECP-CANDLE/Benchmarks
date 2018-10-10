#!/bin/bash -l
#COBALT -A Candle_ECP
#COBALT -n 1 
#COBALT -t 00:30:00
#COBALT -M brettin@anl.gov
#COBALT --jobname Candle-ECP-P1B2 
# #COBALT -q debug

echo $PATH
echo "activiting keras2"
source activate keras2

THIS=$( cd $( dirname $0 ); /bin/pwd )
PYTHONPATH=$PYTHONPATH
PYTHONPATH+=:$THIS:
PYTHONPATH+=:$THIS/../lib
export PYTHONPATH=$PYTHONPATH

echo "$PYTHONPATH"
which python

python /projects/Candle_ECP/brettin/benchmarks/Pilot1/P1B2/p1b2_baseline_keras2.py 

source deactivate

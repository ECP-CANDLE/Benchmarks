#!/bin/bash -l
#COBALT -A Candle_ECP
#COBALT -n 1 
#COBALT -t 01:00:00
#COBALT -M brettin@anl.gov
#COBALT --jobname Candle-ECP 


echo $PATH
echo "activiting Candle_ML"
source activate Candle_ML

THIS=$( cd $( dirname $0 ); /bin/pwd )
PYTHONPATH=$PYTHONPATH
PYTHONPATH+=:$THIS:
PYTHONPATH+=:$THIS/../lib
export PYTHONPATH=$PYTHONPATH

echo "$PYTHONPATH"
which python

python /projects/Candle_ECP/brettin/benchmarks/P1B1/p1b1_baseline.py 

source deactivate

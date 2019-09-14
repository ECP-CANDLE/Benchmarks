#!/bin/bash
#BSUB -W 1:00
#BSUB -nnodes 1
#BSUB -P med106
module load gcc/4.8.5
module load spectrum-mpi/10.3.0.1-20190611
module load cuda/10.1.168
export PATH="/ccs/proj/med106/gounley1/summit/miniconda37/bin:$PATH"
jsrun -n 1 -a 1 -c 42 -g 1 python nt3_baseline_keras2.py --mixed_precision True --config_file nt3_mixed_precision.txt  --gpus 1 -e 3

#!/bin/bash
#BSUB -W 1:00
#BSUB -nnodes 1
#BSUB -P med106
#BSUB -alloc_flags NVME

module load gcc/4.8.5
module load spectrum-mpi/10.3.0.1-20190611
module load cuda/10.1.168
export PATH="/ccs/proj/med106/gounley1/summit/miniconda37/bin:$PATH"


# This is in testing
jsrun -n 1 -a 1 -c 42 -g 6 ./attn_bin_working_jan7_h5.sh 0 > attn1.log 2>&1 &


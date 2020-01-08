#!/bin/bash
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -P med106
module load gcc/4.8.5
module load spectrum-mpi/10.3.0.1-20190611
module load cuda/10.1.168
export PATH="/ccs/proj/med106/gounley1/summit/miniconda37/bin:$PATH"
jsrun -n 1 -a 1 -c 42 -g 1 python uno_baseline_keras2.py --config_file uno_optimized_135v2-a.txt --cache /ccs/home/brettin/project_work/brettin/milestone13/cache/top21_auc --gpus 0 -e 30 --use_exported_data /gpfs/alpine/med106/scratch/hsyoo/Milestone13/top_21_auc_1fold.uno.h5

#!/bin/bash
#BSUB -W 1:00
#BSUB -nnodes 1
#BSUB -P med106
module load gcc/4.8.5
module load spectrum-mpi/10.3.0.1-20190611
module load cuda/10.1.168
export PATH="/ccs/proj/med106/gounley1/summit/miniconda37/bin:$PATH"
jsrun -n 1 -a 1 -c 7 -g 1 python uno_baseline_keras2.py --config_file uno_auc_model.txt --cache $HOME/Benchmarks/cache/top21_auc --gpus 0 -e 3 --use_exported_data $HOME/Benchmarks/top_21_auc_1fold.uno.h5 --save_path $HOME/project_work/save

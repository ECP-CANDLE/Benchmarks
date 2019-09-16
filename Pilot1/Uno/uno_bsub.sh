#!/bin/bash
#BSUB -W 1:00
#BSUB -nnodes 4
#BSUB -P med106
module load gcc/4.8.5
module load spectrum-mpi/10.3.0.1-20190611
module load cuda/10.1.168
export PATH="/ccs/proj/med106/gounley1/summit/miniconda37/bin:$PATH"


# This is the normal keras model
jsrun -n 1 -a 1 -c 42 -g 1 python uno_baseline_keras2.py --config_file uno_auc_model.txt --cache /ccs/home/brettin/project_work/brettin/milestone13/cache/top21_auc --gpus 0 -e 50 --use_exported_data /gpfs/alpine/med106/scratch/brettin/milestone13/top_21_auc_1fold.uno.h5 > keras.model.log 2>&1 &

sleep 2

# This is the tensorflow model
jsrun -n 1 -a 1 -c 42 -g 1 python uno_mixedprecision_tfkeras.py --config_file uno_auc_model.txt --cache /ccs/home/brettin/project_work/brettin/milestone13/cache/top21_auc --gpus 0 -e 50 --use_exported_data /gpfs/alpine/med106/scratch/brettin/milestone13/top_21_auc_1fold.uno.h5 > tensorflow.model.log 2>&1 &

sleep 2

# This is the tensorflow model with mixed-precision
jsrun -n 1 -a 1 -c 42 -g 1 python uno_mixedprecision_tfkeras.py --config_file uno_auc_model.txt --cache /ccs/home/brettin/project_work/brettin/milestone13/cache/top21_auc --gpus 0 -e 50 --use_exported_data /gpfs/alpine/med106/scratch/brettin/milestone13/top_21_auc_1fold.uno.h5 --mixed_precision True > tensorflow.mixedprecision.model.log 2>&1 &

jsrun -n 1 -a 1 -c 42 -g 1 python uno_mixedprecision_tfkeras.py --config_file uno_auc_tensorcore_model.txt --cache /ccs/home/brettin/project_work/brettin/milestone13/cache/top21_auc --gpus 0 -e 50 --use_exported_data /gpfs/alpine/med106/scratch/brettin/milestone13/top_21_auc_1fold.uno.h5 --mixed_precision True > tensorflow.mixedprecision.model.log 2>&1

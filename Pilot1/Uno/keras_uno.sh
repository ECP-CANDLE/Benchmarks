#!/bin/bash
#BSUB -W 1:00
#BSUB -nnodes 3 
#BSUB -P med106
module load gcc/4.8.5
module load spectrum-mpi/10.3.0.1-20190611
module load cuda/10.1.168
export PATH="/ccs/proj/med106/gounley1/summit/miniconda37/bin:$PATH"

echo "start: "$(date)
python uno_baseline_keras2.py --config_file uno_auc_model.txt --cache /ccs/home/brettin/project_work/brettin/milestone13/cache/top21_auc --gpus 3 -e 3 --use_exported_data /gpfs/alpine/med106/scratch/brettin/milestone13/top_21_auc_1fold.uno.h5
echo "stop:  "$(date)

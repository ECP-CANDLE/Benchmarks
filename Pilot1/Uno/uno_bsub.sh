#!/bin/bash
#BSUB -W 1:00
#BSUB -nnodes 2 
#BSUB -P med106
module load gcc/4.8.5
module load spectrum-mpi/10.3.0.1-20190611
module load cuda/10.1.168
export PATH="/ccs/proj/med106/gounley1/summit/miniconda37/bin:$PATH"

# Uno code is located in project_home/Benchmarks on the
# 2020_jan_hackathon git branch.

# Results will be saved for 90 days in member_work/save.
# Inputs are available for 90 days in member_work/splits

input_dir=$HOME/member_work/splits
output_dir=$HOME/member_work/save

for n in $(seq -w 10 15) ; do 
  mkdir -p "$output_dir"/"$n"

  jsrun -n 3 -a 2 -c 14 -g 1 python uno_baseline_keras2.py                        \
	--config_file $HOME/project_home/Benchmarks/Pilot1/Uno/uno_auc_model.txt \
	--cache $HOME/Benchmarks/cache/top21_auc                                 \
	-e 3                                                                     \
	--use_exported_data "$input_dir"/top_21_1fold_"$n".h5                    \
	--save_path "$output_dir"/"$n"

  sleep 2
done


# Notes:
# [2020-01-10 10:52:39 45092] Epoch 0: lr=0.002
# [2020-01-10 10:56:58 45092] [Epoch: 0] loss: 0.540745, mae: 0.088232, r2: -40.810480, val_loss: 0.008663, val_mae: 0.064786, val_r2: 0.606424
# [2020-01-10 10:56:59 45092] Epoch 1: lr=0.00162

# Measured from reporting epoch learn rate:
# 10:56:59 - 10:52:39 = 

# 10:56:59
# 10:52:39
# --------
# 00:04:20 4 minutes 20 seconds per Epoch

# Last FOM (8/23/19) was about 5.74hrs to run 100 epochs. That is 3.44 minutes per epoch.
# Changed from -n 6 -a 1 -c 7 __to__ -n 3 -a 2 -c 14

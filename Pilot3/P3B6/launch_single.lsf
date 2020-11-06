#!/bin/bash
# Begin LSF directives
#BSUB -P med107
#BSUB -J bert_synth
#BSUB -o logs/log.out
#BSUB -W 2:00
#BSUB -nnodes 1
#BSUB -alloc_flags "nvme smt4"

NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
module load gcc
module load open-ce 
module unload darshan

MED106="/gpfs/alpine/proj-shared/med106"
ENV_PATH="$MED106/envs/opence"
WEIGHTS_PATH="$MED106/weights/ncbi_bert_base_pubmed_mimic_uncased"
CMD="python single_node.py --pretrained_weights_path $WEIGHTS_PATH"

conda activate $ENV_PATH

jsrun -n1 -a1 -c7 -g1 --bind=proportional-packed:7 --launch_distribution=packed $CMD > logs/run.log

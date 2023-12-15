#!/bin/bash
#PBS -N st_spe
#PBS -l select=4
#PBS -l walltime=12:00:00
#PBS -q preemptable
#PBS -l filesystems=grand
#PBS -A datascience
#PBS -o logs/
#PBS -e logs/
#PBS -m abe
#PBS -M avasan@anl.gov

module load conda/2022-09-08
conda activate

cd /grand/datascience/avasan/ST_Benchmarks/Test_Tokenizers/SMILESPair_Encoder_continue

NP=16
PPN=4
OUT=logfile.log
let NDEPTH=64/$NP
let NTHREADS=$NDEPTH

TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_ALLOW_GROWTH=true

mpiexec --np 16 -ppn 4 --cpu-bind verbose,list:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 -env NCCL_COLLNET_ENABLE=1 -env NCCL_NET_GDR_LEVEL=PHB python smiles_regress_transformer_run.py > $OUT

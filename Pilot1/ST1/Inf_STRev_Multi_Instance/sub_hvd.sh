#!/bin/bash

#cd /lus/grand/projects/datascience/avasan/ST_Benchmarks/Inference_Runs/Using_Numpy_Data

RANKS_LIST=(32)
#20 24 28 32 36 40 44 48 52)
module load conda/2023-01-10-unstable
conda activate
PWD=/lus/grand/projects/datascience/avasan/ST_Benchmarks/Inference_Runs
DATA_FILE_PATH=${PWD}/Data/BDB
cp -r $DATA_FILE_PATH /dev/shm/

for NRANKS_PER_NODE in "${RANKS_LIST[@]}"
do
    echo quit | nvidia-cuda-mps-control
    rm local_hostfile.01
    ./enable_mps_polaris.sh
        
    NNODES=`wc -l < $PBS_NODEFILE`
    #NRANKS_PER_NODE=$(nvidia-smi -L | wc -l)
    #NRANKS_PER_NODE=32
    let NDEPTH=64/$RANKS_LIST
    let NTHREADS=$NDEPTH
    #cp template_time_file.csv time_info_ranks${NRANKS_PER_NODE}.csv
    
    TF_GPU_ALLOCATOR=cuda_malloc_async
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    
    NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
    echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"
    
    #export CUDA_VISIBLE_DEVICES=0
    
    split --lines=${NNODES} --numeric-suffixes=1 --suffix-length=2 $PBS_NODEFILE local_hostfile.
    
    for lh in local_hostfile*
    do
      echo "Launching mpiexec w/ ${lh}"
      mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --hostfile ${lh} --depth=${NDEPTH}  --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} ./set_affinity_gpu_polaris.sh python smiles_regress_transformer_run.py
#      mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --hostfile ${lh} --depth=${NDEPTH} --cpu-bind verbose,list:0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15:16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31:32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47:48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63 --env OMP_NUM_THREADS=${NTHREADS} ./set_affinity_gpu_polaris.sh python smiles_regress_transformer_run_tfdataset.py
      sleep 1s
    done

    wait
done

#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind numa --env OMP_NUM_THREADS=${NTHREADS} -env OMP_PLACES=threads ./set_affinity_gpu_polaris.sh  python smiles_regress_transformer_run_tfdataset.py
#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind numa --env OMP_NUM_THREADS=${NTHREADS} -env OMP_PLACES=threads ./set_affinity_gpu_polaris.sh  python smiles_regress_transformer_run_tfdataset.py
#mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind core --env OMP_NUM_THREADS=${NTHREADS} -env OMP_PLACES=threads python smiles_regress_transformer_run.py


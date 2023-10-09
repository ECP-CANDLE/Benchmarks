#!/bin/bash -l

export CUDA_MPS_PIPE_DIRECTORY=/dev/shm/CUDA_MPS_PIPE
export CUDA_MPS_LOG_DIRECTORY=/dev/shm/CUDA_MPS_LOG
CUDA_VISIBLE_DEVICES=0,1,2,3 nvidia-cuda-mps-control -d
echo "start_server -uid $( id -u )" | nvidia-cuda-mps-control

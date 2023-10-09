#!/bin/bash

#####################################################################
# These are my own personal directories,
# you will need to change these.
#####################################################################
OUTPUT_DIR=/lus/gila/projects/candle_aesp_CNDA/avasan/Inference_ST_All_Recepts/Inf_STRev_Multi_Instance
WORKDIR=/lus/gila/projects/candle_aesp_CNDA/avasan/Inference_ST_All_Recepts/Inf_STRev_Multi_Instance
cd ${WORKDIR}
 
#####################################################################
# APPLICATION Variables that make a performance difference for tf:
#####################################################################
 
# For most models, channels last is more performance on TF:
DATA_FORMAT="channels_last"
# DATA_FORMAT="channels_first"
 
# Precision for CT can be float32, bfloat16, or mixed (fp16).
PRECISION="float32"
#PRECISION="bfloat16"
# PRECISION="mixed"
 
# Adjust the local batch size:
LOCAL_BATCH_SIZE=1
 
#####################################################################
# FRAMEWORK Variables that make a performance difference for tf:
#####################################################################
 
# Toggle tf32 on (or don't):
ITEX_FP32_MATH_MODE=TF32
#export ITEX_AUTO_MIXED_PERCISION=1
#ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="BFLOAT16"
# unset ITEX_FP32_MATH_MODE
 
# For cosmic tagger, this improves performance:
# (for reference, the default is "setenv ITEX_LAYOUT_OPT \"1\" ")
unset ITEX_LAYOUT_OPT
#unset IPEX_XPU_ONEDNN_LAYOUT_OPT
 
# Set some CCL backend variables:
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ALLREDUCE=topo
export CCL_LOG_LEVEL=warn
export NUMEXPR_MAX_THREADS=208
#208

#####################################################################
# End of perf-adjustment section
#####################################################################
 
 
#####################################################################
# Environment set up, using the latest frameworks drop
#####################################################################
 
# Frameworks have a different oneapi backend at the moment:
#module unload oneapi
#module load frameworks/2023-03-27-experimental-llm_run_mpich
module load frameworks/2023.05.15.001

# activate python environment
source $IDPROOT/bin/activate
conda activate /lus/gila/projects/candle_aesp_CNDA/avasan/conda_tf_env
 
#####################################################################
# End of environment setup section
#####################################################################
 
#####################################################################
# JOB LAUNCH
# Note that this example targets a SINGLE GPU (both tiles)
#####################################################################
 
 
# This string is an identified to store log files:
run_id=sunspot-a21-implicit-gpu-df${DATA_FORMAT}-p${PRECISION}-mb${LOCAL_BATCH_SIZE}-synthetic
 
#####################################################################
# Use this CCS setting to run 4 processes per tile on all 6 GPUs
export TF_ENABLE_ONEDNN_OPTS=0
export ZEX_NUMBER_OF_CCS=0:2,1:2,2:2,3:2,4:2,5:2
#,4:1,5:1
export TOTAL_NUMBER_OF_RANKS=24
export RANKS_PER_NODE=24
export RANKS_PER_TILE=2
#
## launch the job
 
# launch the job
PROCS_PER_TILE=2 mpiexec -np $TOTAL_NUMBER_OF_RANKS -ppn $RANKS_PER_NODE ./gpu_affinity.sh python smiles_regress_transformer_run.py
#mpiexec -np $TOTAL_NUMBER_OF_RANKS -ppn $RANKS_PER_NODE --cpu-bind=verbose,list:0-7,8-15,16-23,24-31,32-39,40-47,48-55,56-63 python test.py \
#framework=tensorflow \
#output_dir=${OUTPUT_DIR}/${run_id} \
#run.id=${run_id} \
#run.compute_mode=XPU \
#run.distributed=True \
#data.data_format=${DATA_FORMAT} \
#run.precision=${PRECISION} \
#run.minibatch_size=${LOCAL_BATCH_SIZE} \
#run.iterations=1

#mpirun -np $TOTAL_NUMBER_OF_RANKS -ppn $RANKS_PER_NODE --cpu-bind=verbose,list:0-7,8-15,16-23,24-31,32-39,40-47,48-55,56-63,64-71,72-79,80-87,88-95 python test_mpi.py



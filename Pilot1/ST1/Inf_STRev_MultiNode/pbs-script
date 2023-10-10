#!/bin/bash
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -q workq
#PBS -A candle_aesp_CNDA
#PBS -m abe
#PBS -M avasan@anl.gov

#
#
# This example will run 12 ranks per node, each rank is binded to a different GPU tile - 12 tiles on a node total
#
#
OUTPUT_DIR=/lus/gila/projects/candle_aesp_CNDA/avasan/DockingSurrogates/Inference/Inference_Scaling/ST_Sort/logs
WORKDIR=/lus/gila/projects/candle_aesp_CNDA/avasan/DockingSurrogates/Inference/Inference_Scaling/ST_Sort/
cd ${WORKDIR}
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
conda activate /lus/gila/projects/candle_aesp_CNDA/avasan/envs/conda_tf_env/

#module restore
module list

echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

NNODES=`wc -l < $PBS_NODEFILE`
export RANKS_PER_NODE=48           # Number of MPI ranks per node
NRANKS=$(( NNODES * RANKS_PER_NODE ))
export PROCS_PER_TILE=4

echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NRANKS}  RANKS_PER_NODE=${RANKS_PER_NODE}"

mpiexec --np ${NRANKS} -ppn ${RANKS_PER_NODE} --cpu-bind verbose,list:0-7:8-15:16-25:26-33:34-41:42-51:52-59:60-67:68-77:78-85:86-93:94-103 \
./set_ze_mask_multiinstance.sh \
python smiles_regress_transformer_run_large.py 

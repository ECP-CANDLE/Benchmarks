#!/bin/sh 

#######################################################################
## Example script for running LBANN autoencoder in LC environment (catalyst)
##
## Script adapted from the LBANN dev team
##
## meant to allow desired hyper paramters as input
##
##
##Notes from source code command line references
##LearnRateMethod = Input("--learning-rate-method", "1 - Adagrad, 2 - RMSprop, 3 - Adam", LearnRateMethod);
##ActivationType = static_cast<activation_type>(Input("--activation-type", "1 - Sigmoid, 2 - Tanh, 3 - reLU, 4 - id", static_cast<int>(ActivationType)));
##
## These hyper parametesr will get converted over to parameters for LBANN command line arguments
## -e -> epoch
## -b -> mini-match
## -a -> activiation type
## -r -> learning rate
## -j -> learning rate decay
## -k -> fraction of training data to use for training
## -g -> dropout rate
## -q -> learning rate method
## -n -> network topology : specify number of nodes in each hidden layer
##
## Example set of hyper parameters to specify 
## params="-e 100 -g 0.1 -b 50 -a 3 -r 0.0001 -j 0.5 -g -1 -q 1 -n 100 -k 0.75 -n 400,300,100"
##
##
#########################################################################

## Need to specify directory location for datafiles
## Original example directory: /p/lscratchf/allen99/anlftp/public/datasets/GDC/data_frames/BySite
## 

## For some reason, this is specfied in two parts 
##LUSTRE_FILEPATH=/p/lscratchf/allen99/anlftp/public/datasets/GDC/data_frames
##DATASET_DIR="BySite"
##
## Now changing this to require input from user on full input and this will be broken into the two parts

## Specifies location of the binary scripts
DIRNAME=/usr/gapps/kpath/lbann/lbannt/lbann/experiments

#Set Script Name variable
SCRIPT=`basename ${0}`

# Figure out which cluster we are on
CLUSTER=`hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g'`

#Initialize variables to default values.
TRAINING_SAMPLES=1
VALIDATION_SAMPLES=1
EPOCHS=10

NETWORK="500x300x100"

PARIO=0
BLOCK_SIZE=256
MODE="false"
MB_SIZE=128
LR=0.0001
ACT=1
LRM=1
PTRS=0.9
TEST_W_TRAIN_DATA=0
LR_DECAY=0.5
DROPOUT=-1
TRAIN_FILE=X
TEST_FILE=X

RUN="srun"

if [ "${CLUSTER}" = "catalyst" ] ; then
#LUSTRE_FILEPATH=/p/lscratchf/allen99/anlftp/public/datasets/GDC/data_frames
ROOT_DATASET_DIR="/l/ssd"
OUTPUT_DIR="/l/ssd/lbann/outputs"
PARAM_DIR="/l/ssd/lbann/models"
ENABLE_HT=
elif [ "${CLUSTER}" = "herd" ]; then
#LUSTRE_FILEPATH=/p/lscratchf/allen99/anlftp/public/datasets/GDC/data_frames
ROOT_DATASET_DIR="/local/ramfs"
OUTPUT_DIR="/local/ramfs/lbann/outputs"
PARAM_DIR="/local/ramfs/lbann/models"
ENABLE_HT=
else
echo "specify host"
exit 0
fi
DATASET_DIR="BySite"
SAVE_MODEL=false
LOAD_MODEL=false
TASKS_PER_NODE=1

USE_LUSTRE_DIRECT=0

#Set fonts for Help.
NORM=`tput sgr0`
#BOLD=`tput bold`
BOLD=
REV=`tput smso`

#Help function
function HELP {
  echo -e \\n"Help documentation for ${BOLD}${SCRIPT}.${NORM}"\\n
  echo -e "${REV}Basic usage:${NORM} ${BOLD}$SCRIPT -t <training set size> -e <epochs> -v <validation set size>${NORM}"\\n
  echo "Command line switches are optional. The following switches are recognized."
  echo "${REV}-a${NORM} <val> --Sets the ${BOLD}activation type${NORM}. Default is ${BOLD}${ACT}${NORM}."
  echo "${REV}-b${NORM} <val> --Sets the ${BOLD}mini-batch size${NORM}. Default is ${BOLD}${MB_SIZE}${NORM}."
  echo "${REV}-c${NORM}       --(CHEAT) Test / validate with the ${BOLD}training data${NORM}. Default is ${BOLD}${TEST_W_TRAIN_DATA}${NORM}."
  echo "${REV}-d${NORM}       --Sets the ${BOLD}debug mode${NORM}."
  echo "${REV}-e${NORM} <val> --Sets the ${BOLD}number of epochs${NORM}. Default is ${BOLD}${EPOCHS}${NORM}."
  echo "${REV}-f${NORM} <val> --Path to the ${BOLD}datasets${NORM}. Default is ${BOLD}${LUSTRE_FILEPATH}${NORM}."
  echo "${REV}-g${NORM} <val> --Sets the ${BOLD}droput rate${NORM}. Default is ${BOLD}${DROPOUT}${NORM}."
  echo "${REV}-i${NORM} <val> --Sets the ${BOLD}parallel I/O limit${NORM}. Default is ${BOLD}${PARIO}${NORM}."
  echo "${REV}-j${NORM} <val> --Sets the ${BOLD}learning rate decay${NORM}. Default is ${BOLD}${LR_DECAY}${NORM}."
  echo "${REV}-k${NORM} <val> --Sets the ${BOLD}percentage of training data to train on${NORM}. Default is ${BOLD}${PTRS}${NORM}."
  echo "${REV}-l${NORM} <val> --Determines if the model is ${BOLD}loaded${NORM}. Default is ${BOLD}${LOAD_MODEL}${NORM}."
  echo "${REV}-m${NORM} <val> --Sets the ${BOLD}mode${NORM}. Default is ${BOLD}${MODE}${NORM}."
  echo "${REV}-n${NORM} <val> --Sets the ${BOLD}network topology${NORM}. Default is ${BOLD}${NETWORK}${NORM}."
  echo "${REV}-o${NORM} <val> --Sets the ${BOLD}output directory${NORM}. Default is ${BOLD}${OUTPUT_DIR}${NORM}."
  echo "${REV}-p${NORM} <val> --Sets the ${BOLD}input parameter directory${NORM}. Default is ${BOLD}${PARAM_DIR}${NORM}."
  echo "${REV}-q${NORM} <val> --Sets the ${BOLD}learning rate method${NORM}. Default is ${BOLD}${LRM}${NORM}."
  echo "${REV}-r${NORM} <val> --Sets the ${BOLD}inital learning rate${NORM}. Default is ${BOLD}${LR}${NORM}."
  echo "${REV}-s${NORM} <val> --Determines if the model is ${BOLD}saved${NORM}. Default is ${BOLD}${SAVE_MODEL}${NORM}."
  echo "${REV}-t${NORM} <val> --Sets the number of ${BOLD}training samples${NORM}. Default is ${BOLD}${TRAINING_SAMPLES}${NORM}."
  echo "${REV}-u${NORM}       --Use the ${BOLD}Lustre filesystem${NORM} directly. Default is ${BOLD}${USE_LUSTRE_DIRECT}${NORM}."
  echo "${REV}-v${NORM} <val> --Sets the number of ${BOLD}validation samples${NORM}. Default is ${BOLD}${VALIDATION_SAMPLES}${NORM}."
  echo "${REV}-x${NORM} <val> --Set ${BOLD}train file name ${NORM}. Default is ${BOLD}${TRAIN_FILE}${NORM}."
  echo "${REV}-y${NORM} <val> --Set ${BOLD}test file name ${NORM}. Default is ${BOLD}${TEST_FILE}${NORM}."
  echo "${REV}-z${NORM} <val> --Sets the ${BOLD}tasks per node${NORM}. Default is ${BOLD}${TASKS_PER_NODE}${NORM}."
  echo -e "${REV}-h${NORM}    --Displays this help message. No further functions are performed."\\n
  exit 1
}

while getopts ":a:b:cde:f:g:hi:j:k:l:m:n:o:p:q:r:s:t:uv:x:y:z:" opt; do
  case $opt in
    a)
      ACT=$OPTARG
      ;;
    b)
      MB_SIZE=$OPTARG
      ;;
    c)
      TEST_W_TRAIN_DATA=1
      ;;
    d)
      RUN="totalview srun -a"
      DEBUGDIR="-debug"
      ;;
    e)
      EPOCHS=$OPTARG
      ;;
    f)
      LUSTRE_FILEPATH=$OPTARG
      ;;
    g)
      DROPOUT=$OPTARG
      ;;
    h)
      HELP
      exit 1
      ;;
    i)
      PARIO=$OPTARG
      ;;
    j)
      LR_DECAY=$OPTARG
      ;;
    k)
      PTRS=$OPTARG
      ;;
    l)
      LOAD_MODEL=$OPTARG
      ;;
    m)
      MODE=$OPTARG
      ;;
    n)
      NETWORK=$OPTARG
      ;;
    o)
      OUTPUT_DIR=$OPTARG
      ;;
    p)
      PARAM_DIR=$OPTARG
      ;;
    q)
      LRM=$OPTARG
      ;;
    r)
      LR=$OPTARG
      ;;
    s)
      SAVE_MODEL=$OPTARG
      ;;
    t)
      TRAINING_SAMPLES=$OPTARG
      ;;
    u)
      USE_LUSTRE_DIRECT=1
      ;;
    v)
      VALIDATION_SAMPLES=$OPTARG
      ;;
    x)
      TRAIN_FILE=$OPTARG
      ;;
    y)
      TEST_FILE=$OPTARG
      ;;
    z)
      TASKS_PER_NODE=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

##LUSTRE_FILEPATH=/p/lscratchf/allen99/anlftp/public/datasets/GDC/data_frames
##DATASET_DIR="BySite"

if [ ! ${LUSTRE_FILEPATH} ] ; then
   HELP
   exit 1
elif [ ! -d ${LUSTRE_FILEPATH} ] ; then
   echo "Error $LUSTRE_FILEPATH should be a directory where the data is located"
   exit 1
else 
   DATASET_DIR=`basename $LUSTRE_FILEPATH`
   LUSTRE_FILEPATH=`dirname $LUSTRE_FILEPATH`
   #echo "check: [$LUSTRE_FILEPATH] [$DATASET_DIR]"
fi


shift $((OPTIND-1))
# now do something with $@

# Look for the binary in the cluster specific build directory
BINDIR="${DIRNAME}/../build/${CLUSTER}.llnl.gov${DEBUGDIR}/model_zoo"

# Once all of the options are parsed, you can setup the environment
source ${DIRNAME}/setup_brain_lbann_env.sh -m mvapich2 -v El_0.86/v86-6ec56a

TASKS=$((${SLURM_NNODES} * ${SLURM_CPUS_ON_NODE}))
if [ ${TASKS} -gt 384 ]; then
TASKS=384
fi
LBANN_TASKS=$((${SLURM_NNODES} * ${TASKS_PER_NODE}))

export PATH=/collab/usr/global/tools/stat/file_bcast/${SYS_TYPE}/fbcast:${PATH}

if [ ${USE_LUSTRE_DIRECT} -eq 1 ]; then

   ROOT_DATASET_DIR=${LUSTRE_FILEPATH}

else

if [ ! -d ${ROOT_DATASET_DIR}/${DATASET_DIR} ]; then
    CMD="pdsh mkdir -p ${ROOT_DATASET_DIR}/${DATASET_DIR}"
    echo "${CMD}"
    ${CMD}
fi

FILES=(${TEST_FILE} ${TRAIN_FILE})
for f in "${FILES[@]}"
do
    FILE=`basename $f`
    if [ ! -e ${ROOT_DATASET_DIR}/${DATASET_DIR}/${FILE} ]; then
        CMD="srun -n${TASKS} -N${SLURM_NNODES} file_bcast_par13 1MB ${LUSTRE_FILEPATH}/${DATASET_DIR}/${f} ${ROOT_DATASET_DIR}/${DATASET_DIR}/${FILE}"
        echo "${CMD}"
        ${CMD}
    fi
done

fi

CMD="${RUN} -N${SLURM_NNODES} -n${LBANN_TASKS} ${ENABLE_HT} --ntasks-per-node=${TASKS_PER_NODE} ${BINDIR}/lbann_greedy_layerwise_autoencoder_nci  --learning-rate ${LR} --activation-type ${ACT} --learning-rate-method ${LRM} --lr-decay-rate ${LR_DECAY} --lambda 0.1 --dataset ${ROOT_DATASET_DIR}/${DATASET_DIR}/ --train-file ${TRAIN_FILE} --test-file ${TEST_FILE} --num-epochs ${EPOCHS} --drop-out ${DROPOUT} --mb-size ${MB_SIZE} --network ${NETWORK} --percentage-training-samples $PTRS" 
echo ${CMD}
${CMD}

#!/bin/bash
set -eu

# UNO TRAIN SH

# arg 1 CUDA_VISIBLE_DEVICES
# arg 2 CANDLE_DATA_DIR
# arg 3 CANDLE_CONFIG

### Path to your CANDLEized model's main Python script###
CANDLE_MODEL=/usr/local/Benchmarks/Pilot1/Uno/uno_baseline_keras2.py

# Make copy of $# before shifts
ARGC=$#

if (( $ARGC < 2 )) ; then
  echo "Uno/train.sh: Illegal number of parameters: given: ${ARGC}"
  echo "CUDA_VISIBLE_DEVICES and CANDLE_DATA_DIR are required"
  exit -1
fi

CUDA_VISIBLE_DEVICES=$1 ; shift
CANDLE_DATA_DIR=$1 ; shift
CANDLE_CONFIG=0

if (( $ARGC == 2 )) ; then
  CMD=( python ${CANDLE_MODEL} )
elif (( $ARGC >= 3 )) ; then
  # If original $3 is a file, set candle_config and passthrough $@
  CANDLE_CONFIG=$1
  if [[ -f $CANDLE_CONFIG ]] ; then
    echo "Uno/train.sh: found CANDLE_CONFIG=$CANDLE_CONFIG"
    shift
    CMD=( python ${CANDLE_MODEL} --config_file $CANDLE_CONFIG $@ )
  else # simply passthrough $@
    CMD=( python ${CANDLE_MODEL} $@ )
  fi
fi

# Report runtime arguments
echo "using CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
echo "using CANDLE_CONFIG ${CANDLE_CONFIG}"

echo "train.sh: running command..."
echo "CMD = ${CMD[@]}"
echo

# Set up environmental variables and execute the model!
if env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
       CANDLE_DATA_DIR=${CANDLE_DATA_DIR} \
       ${CMD[@]}
then
  CODE=0
  echo "train.sh: OK."
else
  CODE=$?
  echo "train.sh: MODEL ERROR: code=$CODE"
fi

exit $CODE

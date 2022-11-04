#!/bin/bash

# arg1 = --in_train
# arg2 = --in_vali
# arg3 = GPU Device

# This scripts runs on 4 gpus. It uses
# PMI RANK to identify the input files.

# This script can be run to train an
# ensemble of models of the same neural
# network architecture and hyperparameters.

# Uses the smiles_regress_transformer.py 
# that instanciates tf.distribute.MirroredStrategy

DEVICES=0,1,2,3
PROG=smiles_regress_transformer.py

EXE="python $PROG"
echo $EXE

readarray -t a < $arg1
base=${a[$PMI_RANK]}
base=$(basename $base)
echo "$(date): rank: $PMI_RANK"
echo "base: $base"

train=${a[$PMI_RANK]}".train"
val=${a[$PMI_RANK]}".val"
echo $train
echo $val

echo "current working dir: $(pwd -P)"
exit
mkdir -p DIR.$base
/bin/cp -n $train DIR.$base/
/bin/cp -n $val DIR.$base/
/bin/cp $PROG DIR.$base

cd DIR.$base/
in_train=$(basename $train)
in_val=$(basename $val)
echo "in_train: $in_train"
echo "in_val: $in_val"

CUDA_VISIBLE_DEVICES=$DEVICES $EXE --ep 4 --in_train $train --in_vali $val > output.log.$$ 2>&1

cd ..
echo "$(date): rank: $PMI_RANK"


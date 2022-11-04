#!/bin/bash

# usage infile
infile=$1
DEVICES=0,1,2,3
PROG=multinode_smiles_regress_transformer.py

EXE="python $PROG"
echo $EXE

# this is for <model>.py with the --in_train --in_vali cli.
# this assumes infile contains the train vali file basename.
# this follows Rick's parallel covid receptor directory scheme.

for n in $(cat $infile) ; do
	echo "$PMI_RANK $(date): $n"

	base=$(basename $n)
	train=$n".train"
	val=$n".val"
	echo "$PMI_RANK in_train $train"
	echo "$PMI_RANK in_val $val"

	echo "$PMI_RANK current working dir: $(pwd -P)"

	DIR=DIR.$PMI_RANK.$base
	mkdir -p $DIR
	/bin/cp -n $train $DIR/
	/bin/cp -n $val $DIR/
	/bin/cp $PROG $DIR/
	/bin/cp pbsutils.py $DIR/

	cd $DIR/
	in_train=$(basename $train)
	in_val=$(basename $val)
	echo "$PMI_RANK in_train: $in_train"
	echo "$PMI_RANK in_val: $in_val"

	echo "$PMI_RANK running CUDA_VISIBLE_DEVICES=$DEVICES $EXE --ep 300 --in_train $train --in_vali $val > output.log.$$ 2>&1"
	CUDA_VISIBLE_DEVICES=$DEVICES $EXE --ep 300 --in_train $train --in_vali $val > output.log.$$ 2>&1
	cd ..
	echo "$PMI_RANK $(date): $n"
done


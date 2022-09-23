#!/bin/bash

# usage infile
infile=$1
DEVICES=0,1,2,3,4,5,6,7
PROG=smiles_regress_transformer.py

EXE="python $PROG"
echo $EXE

for n in $(cat $infile) ; do
	echo "$(date): $n"

	base=$(basename $n)
	train=$n".train"
	val=$n".val"
	echo $train
	echo $val

	echo "current working dir: $(pwd -P)"
	mkdir -p DIR.$base
	/bin/cp -n $train DIR.$base/
	/bin/cp -n $val DIR.$base/
	/bin/cp $PROG DIR.$base

	cd DIR.$base/
	in_train=$(basename $train)
	in_val=$(basename $val)
	echo "in_train: $in_train"
	echo "in_val: $in_val"

	CUDA_VISIBLE_DEVICES=$DEVICES $EXE --ep 300 --in_train $train --in_vali $val > output.log.$$ 2>&1

	cd ..
	echo "$(date): $n"

done


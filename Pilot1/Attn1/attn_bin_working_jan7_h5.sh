#!/bin/bash

prefix="/gpfs/alpine/scratch/brettin/med106"
local_prefix="/mnt/bb/$USER"

m=$1
echo $m

for i in $(cat $m) ; do
  device=$(($i % 6))
  # n="0$i"
  n="00$i"

  export CUDA_VISIBLE_DEVICES=$device
  mkdir -p "$prefix"/save/"$n"
  mkdir -p "$local_prefix"/save/"$n"
  mkdir -p "$local_prefix"/top21_baseline

  echo "copying files to $local_prefix/top21_baseline"

  cp "$prefix"/Data_sets/top21_baseline/top_21_1fold_"$n".h5 \
     $local_prefix/top21_baseline/

  ls $local_prefix/top21_baseline

  echo "running attn_bin_working_jan7_h5.py --in $local_prefix/top21_baseline/top_21_1fold_"$n".h5"
  python attn_bin_working_jan7_h5.py --in $local_prefix/top21_baseline/top_21_1fold_"$n".h5  \
	--ep 2   \
	--save_dir "$local_prefix"/save/"$n"/  > "$local_prefix"/save/"$n".log &
	sleep 2
done

wait

echo "running cp -r $local_prefix/save/* $prefix/save/"
cp -r $local_prefix/save/* $prefix/save/

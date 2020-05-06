#!/bin/bash

prefix="/gpfs/alpine/scratch/brettin/med106"
local_prefix="/mnt/bb/$USER"

m=$1
datadir=$2

echo "input arg file: $m"
echo "input datadir: $datadir"

for i in $(cat $m) ; do

  device=$(($i % 6))

  # pad with zeros to conform to input file names
  if [ $i -lt 10 ] ; then
    n=00"$i"
  else
    n=0"$i"
  fi

  export CUDA_VISIBLE_DEVICES=$device

  # should test if JSM_GPU_ASSIGNMENTS is empty
  if [ $JSM_GPU_ASSIGNMENTS -eq $device ] ; then
    echo "processing line value $i from infile $m using device $device on input $n"
    mkdir -p "$prefix"/save/"$datadir"/"$n"
    mkdir -p "$local_prefix"/save/"$datadir"/"$n"
    mkdir -p "$local_prefix"/"$datadir"

    echo "copying files to $local_prefix/$datadir"
    cp "$prefix"/Data_sets/"$datadir"/top_21_1fold_"$n".h5 \
       $local_prefix/"$datadir"/

    ls $local_prefix/"$datadir"

    echo "running attn_bin_working_jan7_h5.py --in $local_prefix/$datadir/top_21_1fold_"$n".h5"
    python attn_bin_working_jan7_h5.py --in $local_prefix/"$datadir"/top_21_1fold_"$n".h5  \
        --ep 200  \
        --save_dir "$local_prefix"/save/"$datadir"/"$n"/  > "$local_prefix"/save/"$datadir"/"$n".log &
    sleep 2

  fi

done

wait

echo "running cp -r $local_prefix/save/* $prefix/save/"
cp -r $local_prefix/save/* $prefix/save/

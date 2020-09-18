#!/bin/bash

prefix="/ccs/home/brettin/project_work/brettin/adrp/examples/ADRP"
# local_prefix="/mnt/bb/$USER"

m=$1
params=${2:-""}

echo "input arg file: $m"
echo "input params: $params"

for i in $(cat $m) ; do
  OLDIFS=$IFS
  IFS=','
  echo "processing $i"
  read -a strarry <<< "$i"
  IFS=$OLDIFS
  device=$((${strarry[0]} % 6))
  base_name=${strarry[1]}

  export CUDA_VISIBLE_DEVICES=$device

  # should test if JSM_GPU_ASSIGNMENTS is empty
  if [ $JSM_GPU_ASSIGNMENTS -eq $device ] ; then
    echo "processing line value $i from infile $m using device $device on base_name $base_name"

    # set up save path using params
    d=${params//-/}
    d=${d// /_}
    d=${d//\//_}
    d=${d//\./_}

    save_path="$prefix"/save/"$d"/"$base_name"/"${strarry[0]}"
    mkdir -p $save_path

    echo "running ython adrp_baseline_keras2.py --base_name $base_name --output_dir $save_path --save_path $save_path $params"

    python adrp_baseline_keras2.py --base_name $base_name \
	--output_dir $save_path \
        --save_path $save_path  \
	$params
    sleep 2

  fi

done

wait

# echo "running cp -r $local_prefix/save/* $prefix/save/"
# cp -r $local_prefix/save/* $prefix/save/

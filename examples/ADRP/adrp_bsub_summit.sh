#!/bin/bash
#BSUB -W 2:00
#BSUB -nnodes 30
#BSUB -P med106
#BSUB -alloc_flags NVME
#BSUB -J adrp 

module load ibm-wml-ce/1.6.2-3
conda activate /gpfs/alpine/world-shared/med106/sw/condaenv-200408

# hardcoded location of the adrp_x_model.txt file
DIR="/ccs/home/brettin/project_work/brettin/adrp/examples/ADRP"

for i in $(seq 0 4) ; do
  # pad to two digits
  if [ $i -lt 10 ] ; then
    names_file="0""$i"
  else
    names_file="$i"
  fi

  echo "executing jsrun on input file $i"

  params=" --use_sample_weight False --config_file $DIR/adrp_hpo_model.txt "
  jsrun -n 6 -a 1 -c 7 -g 1 ./adrp_baseline_keras2_weighted.sh \
			  "$names_file" "$params" \
                          > attn_bsub.weighted."$names_file"."$LSB_JOBID".log 2>&1 &
  sleep 1

  params=" --use_sample_weight False --config_file $DIR/adrp_default_model.txt "
  jsrun -n 6 -a 1 -c 7 -g 1 ./adrp_baseline_keras2_weighted.sh \
                          "$names_file" "$params" \
                          > attn_bsub.weighted."$names_file"."$LSB_JOBID".log 2>&1 &
  sleep 1

  params=" --use_sample_weights True --sample_weight_type linear --config_file $DIR/adrp_hpo_model.txt "
  jsrun -n 6 -a 1 -c 7 -g 1 ./adrp_baseline_keras2_weighted.sh \
                         "$names_file" "$params" \
                          > attn_bsub.weighted."$names_file"."$LSB_JOBID".log 2>&1 &

  params=" --use_sample_weights True --sample_weight_type linear --config_file $DIR/adrp_default_model.txt "
  jsrun -n 6 -a 1 -c 7 -g 1 ./adrp_baseline_keras2_weighted.sh \
			 "$names_file" "$params"  \
                          > attn_bsub.weighted."$names_file"."$LSB_JOBID".log 2>&1 &
  sleep 1


  params="  --use_sample_weight True --sample_weight_type quadratic --config_file ./adrp_hpo_model.txt "
  jsrun -n 6 -a 1 -c 7 -g 1 ./adrp_baseline_keras2.sh \
                         "$names_file" "$params" \
                          > attn_bsub.weighted."$names_file"."$LSB_JOBID".log 2>&1 &
  sleep 1

  params=" --use_sample_weight True --sample_weight_type quadratic --config_file ./adrp_default_model.txt "
  jsrun -n 6 -a 1 -c 7 -g 1 ./adrp_baseline_keras2.sh \
			 "$names_file" "$params"  \
                          > attn_bsub.weighted."$names_file"."$LSB_JOBID".log 2>&1 &
  sleep 1

done

# -n is number of resource sets to allocate
# -a specifies the number of tasks to start per resource set


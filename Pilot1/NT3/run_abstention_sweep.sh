#!/bin/bash

#vals=0.1
#for filename in /vol/ml/shahashka/temp/Benchmarks/Pilot1/NT3/nt3_cf/noise_both_clusters/nt3.data.threshold.*; do
#    echo $filename
#    python nt3_abstention_keras2.py --noise_cf $filename --output_dir cf_sweep_1104 --run_id $(basename $filename) --epochs 100
#    #cp cf_sweep_0902/EXP000/RUN000/training.log ${filename}_training_0902.log 
#done

for i in $(seq 0 0.1 1); do
    echo $i
    for j in $(seq 1 1 5); do
	python nt3_baseline_keras2.py --label_noise $i --output_dir baseline_label_noise_$i --run_id RUN$j
    done
done

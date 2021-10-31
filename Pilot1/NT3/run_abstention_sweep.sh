#!/bin/bash

vals=0.1
for filename in /vol/ml/shahashka/temp/Benchmarks/Pilot1/NT3/nt3_cf/noise_all_clusters_t=0.1/nt3.data.threshold.*; do
    echo $filename
    python nt3_abstention_keras2.py --noise_cf $filename --output_dir cf_sweep_1030 --run_id $(basename $filename) --epochs 100
    #cp cf_sweep_0902/EXP000/RUN000/training.log ${filename}_training_0902.log 
done

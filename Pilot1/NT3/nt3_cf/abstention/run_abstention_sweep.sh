#!/bin/bash
for filename in /vol/ml/shahashka/xai-geom/nt3/nt3.data*; do
    python nt3_abstention_keras2_cf.py --cf_noise $filename --output_dir cf_sweep_0906 --run_id ${filename:40:21} --epochs 100
    #cp cf_sweep_0902/EXP000/RUN000/training.log ${filename}_training_0902.log 
done

#!/bin/bash

# AUC prediction model
if [ ! -f "top_21_auc_1fold.uno.h5" ]; then
  curl -o top_21_auc_1fold.uno.h5 http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/top_21_auc_1fold.uno.h5
fi

python uno_baseline_keras2.py --config_file uno_auc_model.txt --use_exported_data top_21_auc_1fold.uno.h5 -e 3 --save_weights save/saved.model.weights.h5
python uno_infer.py --data top_21_auc_1fold.uno.h5 \
  --model_file save/'uno.A=relu.B=32.E=3.O=adamax.LR=0.0001.CF=r.DF=d.DR=0.1.wu_lr.re_lr.L1000.D1=1000.D2=1000.D3=1000.D4=1000.D5=1000.FD1=1000.FD2=1000.FD3=1000.FD4=1000.FD5=1000.model.json' \
  --weights_file save/saved.model.weights.h5 \
  --partition val \
  -n 30 \
  --single True \
  --agg_dose AUC

# CLR model
python uno_clr_keras2.py --config_file uno_auc_clr_model.txt --use_exported_data top_21_auc_1fold.uno.h5 -e 3


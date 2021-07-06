#!/bin/bash

python combo_baseline_keras2.py --use_landmark_genes True --warmup_lr True --reduce_lr True -z 256 --epochs 4
python infer.py --sample_set NCIPDM --drug_set ALMANAC --use_landmark_genes -m ./save/combo.A=relu.B=256.E=10.O=adam.LR=None.CF=e.DF=d.wu_lr.re_lr.L1000.D1=1000.D2=1000.D3=1000.model.h5 -w ./save/combo.A=relu.B=256.E=10.O=adam.LR=None.CF=e.DF=d.wu_lr.re_lr.L1000.D1=1000.D2=1000.D3=1000.weights.h5 --epochs 4

# Need to revisit combo_dose.py and infer_dose.py
# python combo_dose.py --use_landmark_genes True -z 256

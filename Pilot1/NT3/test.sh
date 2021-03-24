#!/bin/bash

python nt3_baseline_keras2.py -e 2
# python nt3_baseline_keras2_tensorrt.py -e 2
python nt3_abstention_keras2.py -e 2
CANDLE_DEFAULT_MODEL_FILE=nt3_default_model.txt python nt3_candle_wrappers_baseline_keras2.py -e 2

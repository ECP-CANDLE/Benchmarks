#!/bin/bash

# TRAIN SH
# IMPROVE interface to NT3 model

MODEL=nt3_baseline_keras2.py
MODEL_PATH=`find ${REPO_DIR} -name $MODEL`

echo Train NT3 at ${MODEL_PATH}
echo Options: $@

python $MODEL_PATH $@


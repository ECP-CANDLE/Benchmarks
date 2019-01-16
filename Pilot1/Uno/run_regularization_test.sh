#!/usr/bin/env bash
if [ "$1" != "-p" ]; then
    git pull origin master
fi
rm -r cache
mkdir cache
python uno_baseline_keras2.py --train_sources CCLE --cache cache/CCLE --use_landmark_genes False --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True --gpus 0 1 --epochs 50 --genemania_regularizers True --genemania_filepath ../../../GraphCovTestPMDR/GeneMania_adj.hdf
#!/usr/bin/env bash
if [ "$0" != "-p" ]; then
    git pull origin master
fi

source /raid/aclyde11/gene-graph-conv/graphconv_env/bin/activate
export CUDA_VISABLE_DEVICES='0,1'
rm -r cache
mkdir cache
python uno_baseline_keras2.py --train_sources CCLE --cache cache/CCLE --use_landmark_genes False --preprocess_rnaseq source_scale --no_feature_source True --no_response_source True --gpus 0 1 --epochs 50 --genemania_regularizers True --genemania_filepath ../../../GraphCovTestPMDR/GeneMania_adj.hdf --genemania_regularizers_lam 0.001 --batch_size 1 --growth_bins 4 --single True --test_sources GDSC
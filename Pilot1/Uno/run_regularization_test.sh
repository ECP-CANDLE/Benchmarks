#!/usr/bin/env bash
if [ "$0" != "-p" ]; then
    git pull origin master
fi

source /raid/aclyde11/gene-graph-conv/graphconv_env/bin/activate
export CUDA_VISABLE_DEVICES='0,1'
rm -r cache
mkdir cache
python uno_baseline_keras2.py  --cache cache/CCLE --use_landmark_genes True --tb True --train_sources CCLE CTRP gCSI NCI60 SCL SCLC ALMANAC --test_sources GDSC --preprocess_rnaseq source_scale --workers 12 --no_feature_source True --no_response_source True --gpus 0 1 --epochs 50 --batch_size 5 --residual True --dense_feature_layers_genes 1000 1000 250 --dense_feature_layers 250 --dense 250 250 250  --test_sources GDSC --use_file_rnaseq /home/aclyde11/scratch-area/celsData/combo_snp_rnaseq_stride_data.hdf --by_drug Paclitaxel

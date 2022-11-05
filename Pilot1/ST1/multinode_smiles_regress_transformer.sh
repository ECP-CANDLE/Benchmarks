#!/usr/bin/bash

nvidia-smi -l 3 -f "/lus/eagle/projects/candle_aesp/brettin/Benchmarks/Pilot1/ST1/"$(hostname)".nvivia.log" &
# top -b -d 3 -u brettin > $(hostname).top.log &

ST1="/lus/eagle/projects/candle_aesp/brettin/Benchmarks/Pilot1/ST1/multinode_smiles_regress_transformer.py"

# 2M-flatten
#TRAIN="/lus/eagle/projects/candle_aesp/brettin/2M-flatten/ml.3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.parquet.xform-smiles.csv.reg.train"
#VALI="/lus/eagle/projects/candle_aesp/brettin/2M-flatten/ml.3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.parquet.xform-smiles.csv.reg.val"

# enums
# TRAIN="/lus/eagle/projects/candle_aesp/brettin/enums-test/ml.3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.parquet.xform-smiles.csv.reg.shuf.shuf.train"
# VALI="/lus/eagle/projects/candle_aesp/brettin/enums-test/ml.3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.descriptors.parquet.xform-smiles.csv.reg.shuf.shuf.val"

# enums-test
TRAIN="/lus/eagle/projects/candle_aesp/brettin/enums-test/3.2M.train.csv"
VALI="/lus/eagle/projects/candle_aesp/brettin/enums-test/2M-flatten.3CLPro_7BQY_A_1_F.val"

EP=5
echo "calling python on: $ST1"
echo "traning data: $TRAIN"
echo "validation data: $VALI"
echo "epochs: $EP"

python $ST1 --in_train $TRAIN --in_vali $VALI --ep $EP # > $(hostname).train.log 2>&1

# kill all children in this process group
# trap 'jobs -p | xargs kill' EXIT

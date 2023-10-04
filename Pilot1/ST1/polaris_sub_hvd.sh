#!/bin/bash
#PBS -N st_hvd
#PBS -l select=2
#PBS -l walltime=24:00:00
#PBS -q preemptable
#PBS -l filesystems=grand
#PBS -A datascience
#PBS -o logs/
#PBS -e logs/
#PBS -m abe
#PBS -M avasan@anl.gov

DATA_PATH=/grand/datascience/avasan/ST_Benchmarks/Data/1M-flatten

TFIL=ml.3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.dd.parquet.xform-smiles.csv.reg.train
VFIL=ml.3CLPro_7BQY_A_1_F.Orderable_zinc_db_enaHLL.sorted.4col.dd.parquet.xform-smiles.csv.reg.val

EP=400
NUMHEAD=16
DR_TB=0.1
DR_ff=0.1

ACT=elu
DROP=False
LR=0.0000025
LOSS=mean_squared_error
HVDSWITCH=True

if [$HVDSWITCH = False]; then
    python smiles_regress_transformer_run_hvd.py --in_train ${DATA_PATH}/${TFIL} --in_vali ${DATA_PATH}/${VFIL} --ep $EP --num_heads $NUMHEAD --DR_TB $DR_TB --DR_ff $DR_ff --activation $ACT --drop_post_MHA $DROP --lr $LR --loss_fn $LOSS --hvd_switch $HVDSWITCH

else
    NP=8
    PPN=4
    OUT=logfile.log
    mpiexec --np $NP -ppn $PPN --cpu-bind verbose,list:0,1,2,3,4,5,6,7 -env NCCL_COLLNET_ENABLE=1 -env NCCL_NET_GDR_LEVEL=PHB python smiles_regress_transformer_run_hvd.py  --in_train ${DATA_PATH}/${TFIL} --in_vali ${DATA_PATH}/${VFIL} --ep $EP --num_heads $NUMHEAD --DR_TB $DR_TB --DR_ff $DR_ff --activation $ACT --drop_post_MHA $DROP --lr $LR --loss_fn $LOSS --hvd_switch $HVDSWITCH > $OUT

fi

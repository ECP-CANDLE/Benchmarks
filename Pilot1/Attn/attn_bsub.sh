#!/bin/bash
#BSUB -W 12:00
#BSUB -nnodes 160
#BSUB -P med106
#BSUB -alloc_flags NVME
#BSUB -J attn1

# need 92 nodes for 12 hr run
# with 12 hour run should be able to do 180 (15*12) epochs
#
# at 17 nodes per data set, need to run 6 datasets (102 nodes)
#

module load gcc/4.8.5
module load spectrum-mpi/10.3.0.1-20190611
module load cuda/10.1.168
export PATH="/ccs/proj/med106/gounley1/summit/miniconda37/bin:$PATH"

for i in $(seq 1 16) ; do
  jsrun -n 6 -a 1 -c 7 -g 1 ./attn_bin_working_jan7_h5.sh $i top21_baseline > attn1.top21_baseline."$i".log 2>&1 &
done

for i in $(seq 1 16) ; do
  jsrun -n 6 -a 1 -c 7 -g 1 ./attn_bin_working_jan7_h5.sh $i top21_r.0_baseline > attn1.top21_r.0_baseline."$i".log 2>&1 &
done

for i in $(seq 1 16) ; do
  jsrun -n 6 -a 1 -c 7 -g 1 ./attn_bin_working_jan7_h5.sh $i top21_r.0_gap1 > attn1.top21_r.0_gap1."$i".log 2>&1 &
done

for i in $(seq 1 16) ; do
  jsrun -n 6 -a 1 -c 7 -g 1 ./attn_bin_working_jan7_h5.sh $i top21_r.0_gap2 > attn1.top21_r.0_gap2."$i".log 2>&1 &
done

for i in $(seq 1 16) ; do
  jsrun -n 6 -a 1 -c 7 -g 1 ./attn_bin_working_jan7_h5.sh $i top21_r.5_baseline > attn1.top21_r.5_baseline."$i".log 2>&1 &
done

for i in $(seq 1 16) ; do
  jsrun -n 6 -a 1 -c 7 -g 1 ./attn_bin_working_jan7_h5.sh $i top21_r.5_gap1 > attn1.top21_r.5_gap1."$i".log 2>&1 &
done

for i in $(seq 1 16) ; do
  jsrun -n 6 -a 1 -c 7 -g 1 ./attn_bin_working_jan7_h5.sh $i top21_r.5_gap2 > attn1.top21_r.5_gap2."$i".log 2>&1 &
done

for i in $(seq 1 16) ; do
  jsrun -n 6 -a 1 -c 7 -g 1 ./attn_bin_working_jan7_h5.sh $i top21_r.9_baseline > attn1.top21_r.9_baseline."$i".log 2>&1 &
done

for i in $(seq 1 16) ; do
  jsrun -n 6 -a 1 -c 7 -g 1 ./attn_bin_working_jan7_h5.sh $i top21_r.9_gap1 > attn1.top21_r.9_gap1."$i".log 2>&1 &
done

for i in $(seq 1 16) ; do
  jsrun -n 6 -a 1 -c 7 -g 1 ./attn_bin_working_jan7_h5.sh $i top21_r.9_gap2 > attn1.top21_r.9_gap2."$i".log 2>&1 &
done

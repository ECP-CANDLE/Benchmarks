#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=4:system=polaris
#PBS -l place=scatter
#PBS -N srt-eagle
#PBS -A CSC249ADOA01
#PBS -q debug-scaling

echo "PBS_O_WORKDIR: $PBS_O_WORKDIR"
echo "nodes: "
cat $PBS_NODEFILE

module load conda/2022-07-19
conda activate base

cd $PBS_O_WORKDIR
mpiexec -ppn 1 -n 4 $PBS_O_WORKDIR/run_train.sh $arg1

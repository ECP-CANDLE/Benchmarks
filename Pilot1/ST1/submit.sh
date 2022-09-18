#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=10:system=polaris
#PBS -l place=scatter
#PBS -N srt-eagle
#PBS -A CSC249ADOA01
#PBS -q debug-scaling

if [ -z "$arg1" ]; then
        echo "arg1 not set, it should be a filename"
        exit
fi

echo "PBS_O_WORKDIR: $PBS_O_WORKDIR"
echo "nodes: "
cat $PBS_NODEFILE

module load conda/2022-07-19
conda activate base

# the absolute path of the current working directory of the qsub command
cd $PBS_O_WORKDIR

# run one run_train.sh process per node on each of 10 nodes
mpiexec -ppn 1 -n 10 $PBS_O_WORKDIR/run_train.sh $arg1

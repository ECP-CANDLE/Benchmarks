#PBS -l walltime=01:00:00
#PBS -l select=32:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=eagle:home
#PBS -N mnsrt_32
#PBS -A CSC249ADOA01
#PBS -q prod

module load conda
conda activate base

mpiexec -n 32 /lus/eagle/projects/candle_aesp/brettin/Benchmarks/Pilot1/ST1/multinode_smiles_regress_transformer.sh

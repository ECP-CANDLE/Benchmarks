awk '{print $0, "slots=12"}' $PBS_NODEFILE > hostfile_mpi
awk '{print $0}' $PBS_NODEFILE > hostfile_mpis

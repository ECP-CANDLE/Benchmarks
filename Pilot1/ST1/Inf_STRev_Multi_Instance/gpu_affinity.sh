#!/usr/bin/env bash
​
display_help() {
  echo " Will map gpu tile to rank in compact and then round-robin fashion"
  echo " Usage:"
  echo "   mpiexec -np N gpu_tile_compact.sh ./a.out"
  echo
  echo " Example 3 GPU of 2 Tiles with 7 Ranks:"
  echo "   0 Rank 0.0"
  echo "   1 Rank 0.1"
  echo "   2 Rank 1.0"
  echo "   3 Rank 1.1"
  echo "   4 Rank 2.0"
  echo "   5 Rank 2.1"
  echo "   6 Rank 0.0"
  echo 
  echo " Hacked together by apl@anl.gov, please contact if bug found"
  exit 1
}
​
#This give the exact GPU count i915 knows about and I use udev to only enumerate the devices with physical presence.
num_gpu=$(/usr/bin/udevadm info /sys/module/i915/drivers/pci:i915/* |& grep -v Unknown | grep -c "P: /devices")
num_tile=2
​
if [[ -v PROCS_PER_TILE ]]; then
    _PROCS_PER_TILE=$PROCS_PER_TILE
    _PROCS_PER_GPU=$((PROCS_PER_TILE*2))
fi
​
if [[ -v PROCS_PER_GPU ]]; then
    _PROCS_PER_GPU=$PROCS_PER_GPU
    if [[ $_PROCS_PER_GPU -gt $((_PROCS_PER_TILE*2)) ]]; then
       echo "PROCS_PER_GPU cannot be greater than 2*PROCS_PER_TILE"
       exit 1
    fi
fi
​
if [ "$#" -eq 0 ] || [ "$1" == "--help" ] || [ "$1" == "-h" ] || [ "$num_gpu" = 0 ] ; then
  display_help
fi
​
# Get the RankID from different launcher
if [[ -v MPI_LOCALRANKID ]]; then
  _MPI_RANKID=$MPI_LOCALRANKID 
elif [[ -v PALS_LOCAL_RANKID ]]; then
  _MPI_RANKID=$PALS_LOCAL_RANKID
else
  display_help
fi
​
if [[ $_MPI_RANKID -eq 0 ]]; then
    echo "Number of GPUs: $num_gpu"
fi
if [[ $_PROCS_PER_GPU -eq 1 ]]; then
    gpu_id=$((_MPI_RANKID % num_gpu ))
    tile_id=0
elif [[ -v _PROCS_PER_TILE ]]; then
    gpu_id=$(((_MPI_RANKID / $_PROCS_PER_GPU) % num_gpu))
    if [[ $_PROCS_PER_TILE -eq $_PROCS_PER_GPU ]]; then
        tile_id=0
    else
        tile_id=$(((_MPI_RANKID / $_PROCS_PER_TILE) % num_tile))
    fi
else
    gpu_id=$(((_MPI_RANKID / num_tile) % num_gpu))
    tile_id=$((_MPI_RANKID % num_tile))
fi
unset EnableWalkerPartition
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK=$gpu_id.$tile_id
echo "AFFINITY_MASK [$_MPI_RANKID]: $ZE_AFFINITY_MASK"
#https://stackoverflow.com/a/28099707/7674852
"$@"

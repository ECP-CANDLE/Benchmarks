prefix=/scratch/brettin/Agg_attn_bin_iter1
# prefix=$HOME

for m in $(seq -w 0 7); do

  device=$(($m % 8))
  n="00$m"

  export CUDA_VISIBLE_DEVICES=$device
  mkdir -p $prefix/save/$n

  python attn_bin_working_jan7_h5.py --in /scratch/data/benchmarks/binary_811_splits/top_21_1fold_"$n".h5  \
	--ep 200   \
	--save_dir $prefix/save/"$n"/  > $prefix/save/$n.log &

  sleep 2
done

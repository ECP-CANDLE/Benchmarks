prefix=/scratch/brettin/Agg_attn_bin_iter2

m=$1
echo $m

device=$(($m % 8))
n="0$m"

export CUDA_VISIBLE_DEVICES=$device
mkdir -p $prefix/save/$n

python attn_bin_working_jan7_h5.py --in $prefix/top21_r10/top_21_1fold_"$n".h5  \
	--ep 200   \
	--save_dir $prefix/save/"$n"/  > $prefix/save/$n.log

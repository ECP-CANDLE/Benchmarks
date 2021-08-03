# it looks like this script starts at training size equal to DATAPOINTS
# and increases by adding DATAPOINTS to training size on each iteration.

epochs=$1

prefix=./$$

DATA_POINTS=30000

for m in $(seq -w 0 7); do

  device=$(($m % 8))
  n="00$m"

  export CUDA_VISIBLE_DEVICES=$device
  mkdir -p $prefix/save/$n

  training_size=$(($DATA_POINTS*($m+1)))
  echo "python ./attn_activations_tf2.py --training_size $training_size -s $prefix/save/$n --cca_epsilon 1e-7 --epochs $epochs \> $prefix/save/$n.log"
  python ./attn_activations_tf2.py \
	  --training_size $training_size \
	  -s $prefix/save/$n/  \
	  --cca_epsilon 1e-7  \
	  --epochs $epochs \
	  > $prefix/save/$n.log 2>&1 &

  sleep 2
done

wait

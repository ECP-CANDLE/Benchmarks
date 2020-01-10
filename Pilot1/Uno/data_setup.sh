
export PATH="/ccs/proj/med106/gounley1/summit/miniconda37/bin:$PATH"

for n in $(seq -w 0 10) ; do
   r=$RANDOM
   mkdir -p $n
   pushd $n
   cp $HOME/Benchmarks/splits/"1fold_s"$n* .

   # Generate h5 train validation splits
   # python ../topN_to_uno.py \
   #      --dataframe_from $HOME/Benchmarks/top_21.res_reg.cf_rnaseq.dd_dragon7.labeled.hdf5 \
   #      --fold 1fold_$n \
   #      --output top_21_1fold_$n.h5

   # TODO Generate tfrecord files
       
   popd
done


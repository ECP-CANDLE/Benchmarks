
for n in $(seq -w 0 99) ; do
   r=$RANDOM
   mkdir -p $n
   pushd $n
   # Generate 1fold_$n_tr_id.csv and 1fold_$n_vl_id.csv
   python data_split.py --seed $r --output 1fold_$n

  # Generate h5 train validation splits
  python topN_to_uno.py \
       --dataframe_from ../top_21.res_reg.cf_rnaseq.dd_dragon7.labeled.hdf5 \
       --fold 1fold_$n \
       --output top_21_1fold_$n.h5

   # TODO Generate tfrecord files
       
   popd
done


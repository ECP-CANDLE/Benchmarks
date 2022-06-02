
for n in $(cat $1) ; do
	echo $n
	./attn_bin_working_jan7_h5.sh $n
done

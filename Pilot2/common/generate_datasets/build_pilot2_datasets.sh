#!/bin/bash

# Typically this might be ./build_pilot2_datasets.sh /p/lscratche/timcar /p/lscratche/krasu
SRCDIR=$1
shift
DSTPATH=$1
echo "reading from $SRCDIR to $DSTPATH"

dir3k1="${SRCDIR}/RAS/3-component-system/DPPC-DOPC-CHOL/af-restraints-290k/simulations/3k-lipids-290k/run10/*us.*-center-mol.xtc"
dir3k2="${SRCDIR}/RAS/3-component-system/DPPC-DOPC-CHOL/af-restraints-290k/simulations/3k-lipids-290k/run32/*us.*-center-mol.xtc"
dir3k3="${SRCDIR}/RAS/3-component-system/DPPC-DOPC-CHOL/af-restraints-290k/simulations/3k-lipids-290k/run43/*us.*-center-mol.xtc"
dir6k1="${SRCDIR}/RAS/3-component-system/DPPC-DOPC-CHOL/af-restraints-290k/6k-lipids/simulations/6k-lipids-290k/run10/*us.*-center-mol.xtc"
dir6k2="${SRCDIR}/RAS/3-component-system/DPPC-DOPC-CHOL/af-restraints-290k/6k-lipids/simulations/6k-lipids-290k/run32/*us.*-center-mol.xtc"
dir6k3="${SRCDIR}/RAS/3-component-system/DPPC-DOPC-CHOL/af-restraints-290k/6k-lipids/simulations/6k-lipids-290k/run43/*us.*-center-mol.xtc"

DATE=`date +%m%d%Y`
CANDLEDIR="CANDLE_benchmarks_$DATE"
DSTDIR="${DSTPATH}/${CANDLEDIR}"

mkdir -p "${DSTPATH}"

for i in $dir3k1 $dir3k2 $dir3k3 $dir6k1 $dir6k2 $dir6k3 
do
	if [ "$i" == $dir3k1 ] ; then hh="3k_run10"; fi
	if [ "$i" == $dir3k2 ] ; then hh="3k_run32"; fi
	if [ "$i" == $dir3k3 ] ; then hh="3k_run43"; fi
	if [ "$i" == $dir6k1 ] ; then hh="6k_run10"; fi
	if [ "$i" == $dir6k2 ] ; then hh="6k_run32"; fi
	if [ "$i" == $dir6k3 ] ; then hh="6k_run43"; fi
	echo hh "$hh"
	hx=`basename $i .xtc`
	hy=`echo "$hh"_"$hx"`
	j=`echo $i | sed "s/.xtc//"`
	m=`echo $i | sed "s/-center-mol.xtc//"`
	k=`echo $i | sed 's=../==' | sed 's/.xtc//' | sed "s/\//_/g"| sed 's/\//_/'`
	dest=${DSTDIR}"/"$hy".dir"
	mkdir -p "$dest"
	chown ${USER}:kras $dest
	chmod 775 $dest	
	echo python gen_pilot2_dataset_from_gromacs_xtc.py -s "$m".tpr -f "$j".xtc -o "$dest"/"$hy"_chunk_
	python gen_pilot2_dataset_from_gromacs_xtc.py -s "$m".tpr -f "$j".xtc -o "$dest"/"$hy"_chunk_
	chown -R ${USER}:kras $dest"/../"
	chmod -R 775 $dest"/../"
done

#!/bin/bash

# Typically this might be ./build_pilot2_datasets.sh /p/lscratche/timcar /p/lscratche/krasu
SRCDIR=$1
shift
DSTPATH=$1
echo "reading from $SRCDIR to $DSTPATH"

dir3k1="RAS/3-component-system/DPPC-DOPC-CHOL/af-restraints-290k/simulations/3k-lipids-290k/run10"
dir3k2="RAS/3-component-system/DPPC-DOPC-CHOL/af-restraints-290k/simulations/3k-lipids-290k/run32"
dir3k3="RAS/3-component-system/DPPC-DOPC-CHOL/af-restraints-290k/simulations/3k-lipids-290k/run43"
dir3k4="3-component-systems/DPPC-DIPC-CHOL/restraints-antifreeze/simulations/3k-lipids-297k/run16/"
dir6k1="RAS/3-component-system/DPPC-DOPC-CHOL/af-restraints-290k/6k-lipids/simulations/6k-lipids-290k/run10"
dir6k2="RAS/3-component-system/DPPC-DOPC-CHOL/af-restraints-290k/6k-lipids/simulations/6k-lipids-290k/run32"
dir6k3="RAS/3-component-system/DPPC-DOPC-CHOL/af-restraints-290k/6k-lipids/simulations/6k-lipids-290k/run43"
ADJ=
traj_glob="*us.*.tpr"

TMPDIR="/l/ssd"
#TMPDIR="/tmp/vanessen"

DATE=`date +%m%d%Y`
CANDLEDIR="CANDLE_benchmarks_$DATE"
DSTDIR="${DSTPATH}/${CANDLEDIR}"

mkdir -p "${DSTPATH}"

for i in $dir3k4 #$dir3k1 $dir3k2 $dir3k3 $dir6k1 $dir6k2 $dir6k3
do
    if [ "$i" == $dir3k1 ] ; then hh="3k_run10"; fi
    if [ "$i" == $dir3k2 ] ; then hh="3k_run32"; fi
    if [ "$i" == $dir3k3 ] ; then hh="3k_run43"; fi
    if [ "$i" == $dir3k4 ] ; then hh="3k_run16"; fi
    if [ "$i" == $dir6k1 ] ; then hh="6k_run10"; fi
    if [ "$i" == $dir6k2 ] ; then hh="6k_run32"; ADJ="skip-first-50ns."; fi
    if [ "$i" == $dir6k3 ] ; then hh="6k_run43"; ADJ="skip-first-50ns."; fi
    mkdir -p ${TMPDIR}/${i}
    glob="*us.*-center-mol.${ADJ}xtc"
    `cp -r ${SRCDIR}/${i}/${glob} ${TMPDIR}/${i}`
    `cp -r ${SRCDIR}/${i}/${traj_glob} ${TMPDIR}/${i}`
    echo hh "$hh"
    hx=`basename ${TMPDIR}/$i/${glob} .xtc`
    hx=`echo $hx | sed "s/-center-mol//"`
    hy=`echo "$hh"_"$hx"`
    j=`echo ${TMPDIR}/$i/${glob} | sed "s/.xtc//"`
    m=`echo ${TMPDIR}/$i/${glob} | sed "s/-center-mol.${ADJ}xtc//"`
    k=`echo ${TMPDIR}/$i/${glob} | sed 's=../==' | sed 's/.xtc//' | sed "s/\//_/g"| sed 's/\//_/'`
    dest=${DSTDIR}"/"$hy".dir"
    mkdir -p "$dest"
    chown ${USER}:kras $dest
    chmod 775 $dest
    echo python gen_pilot2_dataset_from_gromacs_xtc.py -s "$m".tpr -f "$j".xtc -o "$dest"/"$hy"_chunk_
    python gen_pilot2_dataset_from_gromacs_xtc.py -s "$m".tpr -f "$j".xtc -o "$dest"/"$hy"_chunk_
    chown -R ${USER}:kras $dest"/../"
    chmod -R 775 $dest"/../"
done

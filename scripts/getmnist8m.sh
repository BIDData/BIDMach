#!/bin/bash
# run this to load the MNIST8M data

BIDMACH_SCRIPTS="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_SCRIPTS=`readlink -f "${BIDMACH_SCRIPTS}"`
  export WGET='wget --no-check-certificate'
else 
  while [ -L "${BIDMACH_SCRIPTS}" ]; do
    BIDMACH_SCRIPTS=`readlink "${BIDMACH_SCRIPTS}"`
  done
  export WGET='curl --retry 20 -O'
fi
export BIDMACH_SCRIPTS=`dirname "$BIDMACH_SCRIPTS"`
cd ${BIDMACH_SCRIPTS}
BIDMACH_SCRIPTS=`pwd`
BIDMACH_SCRIPTS="$( echo ${BIDMACH_SCRIPTS} | sed 's+/cygdrive/\([a-z]\)+\1:+' )" 


echo "Loading MNIST8M data"

MNIST8M="${BIDMACH_SCRIPTS}/../data/MNIST8M"
mkdir -p ${MNIST8M}/parts
cd ${MNIST8M}

if [ ! -e mnist8m.bz2 ]; then
    ${WGET} http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist8m.bz2
fi

bunzip2 -c mnist8m.bz2 > mnist8m.libsvm

if [ ! `uname` = "Darwin" ]; then
    split -l 100000 -d mnist8m.libsvm parts/part
else
    split -l 100000 mnist8m.libsvm parts/part
    j=0
    for i in {a..z}{a..z}; do
	jj=`printf "%02d" $j`
	mv parts/part$i parts/part$jj
	j=$((j+1))
	if [ $j -gt 80 ]; then break; fi
    done
fi

cd ${MNIST8M}/parts
${BIDMACH_SCRIPTS}/../bidmach ${BIDMACH_SCRIPTS}/processmnist8m.ssc

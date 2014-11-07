#!/bin/bash

BIDMACH_SCRIPTS="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_SCRIPTS=`readlink -f "${BIDMACH_SCRIPTS}"`
else 
  while [ -L "${BIDMACH_SCRIPTS}" ]; do
    BIDMACH_SCRIPTS=`readlink "${BIDMACH_SCRIPTS}"`
  done
  alias wget='curl --retry 20 -O'
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
    wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist8m.bz2
fi

bunzip2 -c mnist8m.bz2 > mnist8m.lsvm

split -l 100000 -d mnist8m.lsvm parts/part

cd ${MNIST8M}/parts
../../../bidmach '../../../scripts/processmnist8m.ssc'
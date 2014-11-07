#!/bin/bash

BIDMACH_SCRIPTS="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_SCRIPTS=`readlink -f "${BIDMACH_SCRIPTS}"`
else 
  while [ -L "${BIDMACH_SCRIPTS}" ]; do
    BIDMACH_SCRIPTS=`readlink "${BIDMACH_SCRIPTS}"`
  done
  alias wget='curl --retry 2 -O'
fi
export BIDMACH_SCRIPTS=`dirname "$BIDMACH_SCRIPTS"`
cd ${BIDMACH_SCRIPTS}
BIDMACH_SCRIPTS=`pwd`
BIDMACH_SCRIPTS="$( echo ${BIDMACH_SCRIPTS} | sed 's+/cygdrive/\([a-z]\)+\1:+' )" 


echo "Loading MNIST data"

MNIST="${BIDMACH_SCRIPTS}/../data/MNIST"
mkdir -p ${MNIST}
cd ${MNIST}

if [ ! -e mnist.bz2 ]; then
    wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2
fi
if [ ! -e mnist.t.bz2 ]; then
    wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.t.bz2
fi

bunzip2 -c mnist.bz2 > mnist.lsvm
bunzip2 -c mnist.t.bz2 > mnist.t.lsvm

../../bidmach '../../scripts/processmnist.ssc'

rm mnist.lsvm
rm mnist.t.lsvm
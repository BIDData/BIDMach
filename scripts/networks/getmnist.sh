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

MNIST="${BIDMACH_SCRIPTS}/../../data/MNIST"
mkdir -p ${MNIST}
cd ${MNIST}

if [ ! -e train-images-idx3-ubyte ]; then
    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz 
    gunzip train-images-idx3-ubyte.gz
fi
if [ ! -e train-labels-idx1-ubyte ]; then
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    gunzip train-labels-idx1-ubyte.gz
fi

if [ ! -e t10k-images-idx3-ubyte ]; then
    wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz 
    gunzip t10k-images-idx3-ubyte.gz
fi
if [ ! -e t10k-labels-idx1-ubyte ]; then
    wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    gunzip t10k-labels-idx1-ubyte.gz
fi

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
cd "${BIDMACH_SCRIPTS}"
BIDMACH_SCRIPTS=`pwd`
BIDMACH_SCRIPTS="$( echo ${BIDMACH_SCRIPTS} | sed 's+/cygdrive/\([a-z]\)+\1:+' )" 

echo "Loading CIFAR100 data"

CIFAR100="${BIDMACH_SCRIPTS}/../../data/CIFAR100"
mkdir -p ${CIFAR100}/parts
cd ${CIFAR100}

if [ ! -e t10k-labels-idx1-ubyte ]; then
    wget http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
    tar -xf cifar-100-binary.tar.gz 
    rm -f cifar-100-binary.tar.gz
    mv cifar-100-binary/* . 
    rm -rf cifar-100-binary
fi

echo "Processing CIFAR100 data"
cd "${BIDMACH_SCRIPTS}"
../../bidmach processcifar100.ssc

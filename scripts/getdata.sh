#!/bin/bash

BIDMACH_ROOT="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_ROOT=`readlink -f "${BIDMACH_ROOT}"`
else
  while [ -L "${BIDMACH_ROOT}" ]; do
    BIDMACH_ROOT=`readlink "${BIDMACH_ROOT}"`
  done
fi
BIDMACH_ROOT=`dirname "${BIDMACH_ROOT}"`
BIDMACH_ROOT=`cd ${BIDMACH_ROOT}/..;pwd -P`
BIDMACH_ROOT="$( echo ${BIDMACH_ROOT} | sed 's+/cygdrive/\(.\)+\1:+' )"

cd "${BIDMACH_ROOT}/scripts"

./getrcv1.sh

./getuci.sh nips

./getuci.sh nytimes

./getdigits.sh

./getmovies.sh

./getmnist.sh

./getcifar10.sh

./getmnist8m.sh

# this one is huge, make sure you really want it
# ./getuci.sh pubmed




#!/bin/bash

source=$1

BIDMACH_ROOT="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_ROOT=`readlink -f "${BIDMACH_ROOT}"`
else 
  while [ -L "${BIDMACH_ROOT}" ]; do
    BIDMACH_ROOT=`readlink "${BIDMACH_ROOT}"`
  done
fi
BIDMACH_ROOT=`dirname "$BIDMACH_ROOT"`
BIDMACH_ROOT=`pwd`
BIDMACH_ROOT="$( echo ${BIDMACH_ROOT} | sed s+/cygdrive/c+c:+ )" 

cp ${source}/lib/*.jar ${BIDMACH_ROOT}/lib
cp ${source}/lib/*.so ${BIDMACH_ROOT}/lib
cp ${source}/lib/*.dll ${BIDMACH_ROOT}/lib
cp ${source}/lib/*.dylib ${BIDMACH_ROOT}/lib
cp ${source}/lib/*.jnilib ${BIDMACH_ROOT}/lib

cp ${source}/BIDMach.jar ${BIDMACH_ROOT}

mkdir -p ${BIDMACH_ROOT}/cbin
cp ${source}/cbin/* ${BIDMACH_ROOT}/cbin


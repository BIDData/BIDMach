#!/bin/bash

BIDMACH_ROOT="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_ROOT=`readlink -f "${BIDMACH_ROOT}"`
else 
  while [ -L "${BIDMACH_ROOT}" ]; do
    BIDMACH_ROOT=`readlink "${BIDMACH_ROOT}"`
  done
fi
BIDMACH_ROOT=`dirname "$BIDMACH_ROOT"`
pushd "${BIDMACH_ROOT}"  > /dev/null
BIDMACH_ROOT=`pwd`
BIDMACH_ROOT="$( echo ${BIDMACH_ROOT} | sed s+/cygdrive/c+c:+ )" 

if [ `uname` = "Darwin" ]; then
    binnames=".dylib,.jnilib"
    cdir="osx"
elif [ "$OS" = "Windows_NT" ]; then
    binnames=".dll"
    cdir="win"
else 
    binnames=".so"
    cdir="linux"
fi

source="http://www.cs.berkeley.edu/~jfc/biddata"

cd ${BIDMACH_ROOT}/lib
wget -r -A.txt,.html,.jar ${source}/lib/
wget -r -A${binnames} ${source}/lib/

mkdir -p ${BIDMACH_ROOT}/cbin
cd ${BIDMACH_ROOT}/cbin
wget -r -A.exe ${source}/cbin/${cdir}/

mv ${BIDMACH_ROOT}/lib/BIDMach.jar ${BIDMACH_ROOT}


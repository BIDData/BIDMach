#!/bin/bash
if [[ ${ARCH} == "" ]]; then
    ARCH=`arch`
fi

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

source="http://www.cs.berkeley.edu/~jfc/biddata"
cd ${BIDMACH_ROOT}/lib

if [ `uname` = "Darwin" ]; then
    subdir="osx"
    suffix="dylib"
    curl -o liblist.txt ${source}/lib/liblist_osx.txt 
elif [ "$OS" = "Windows_NT" ]; then
    subdir="win"
    suffix="dll"
    curl -o liblist.txt ${source}/lib/liblist_win.txt
else
    if [[ "${ARCH}" == arm* || "${ARCH}" == aarch* ]]; then
        subdir="linux_arm"
	suffix="so"
        curl -o liblist.txt ${source}/lib/liblist_linux_arm.txt
    else
        subdir="linux"
	suffix="so"
        curl -o liblist.txt ${source}/lib/liblist_linux.txt
    fi
fi

while read fname; do
   echo -e "\nDownloading ${fname}"
   curl --retry 2  -z ${fname} -o ${fname} ${source}/lib/${fname}
   chmod 755 ${fname}
done < liblist.txt

mkdir -p ${BIDMACH_ROOT}/cbin
cd ${BIDMACH_ROOT}/cbin
curl -o exelist.txt ${source}/cbin/exelist.txt

while read fname; do
    echo -e "\nDownloading ${fname}"
    curl --retry 2 -o ${fname} ${source}/cbin/${subdir}/${fname}
    chmod 755 ${fname}
done < exelist.txt

chmod 755 ${BIDMACH_ROOT}/cbin/*

mv ${BIDMACH_ROOT}/lib/BIDMach.jar ${BIDMACH_ROOT}

cd ${BIDMACH_ROOT}
libs=`echo lib/*bidmat*.${suffix} lib/*iomp5*.${suffix}`
echo "Packing native libraries in the BIDMat jar"
jar uvf lib/BIDMat.jar $libs

rm -f ${BIDMACH_ROOT}/src/main/resources/lib/*.${suffix} 
cp ${BIDMACH_ROOT}/lib/*bidmach*.${suffix} ${BIDMACH_ROOT}/src/main/resources/lib
cp ${BIDMACH_ROOT}/lib/*iomp5*.${suffix} ${BIDMACH_ROOT}/src/main/resources/lib
cd ${BIDMACH_ROOT}/src/main/resources
libs=`echo lib/*.${suffix}`
cd ${BIDMACH_ROOT}
echo "Packing native libraries in the BIDMach jar"
jar uvf BIDMach.jar $libs



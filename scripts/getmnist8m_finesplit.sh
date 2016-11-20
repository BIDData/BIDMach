#!/bin/bash
# run this to load the MNIST8M data

BIDMACH_SCRIPTS="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_SCRIPTS=`readlink -f "${BIDMACH_SCRIPTS}"`
  export WGET='wget -c --no-check-certificate'
else
  while [ -L "${BIDMACH_SCRIPTS}" ]; do
    BIDMACH_SCRIPTS=`readlink "${BIDMACH_SCRIPTS}"`
  done
  export WGET='curl -C - --retry 20 -O'
fi
export BIDMACH_SCRIPTS=`dirname "$BIDMACH_SCRIPTS"`
cd ${BIDMACH_SCRIPTS}
BIDMACH_SCRIPTS=`pwd`
BIDMACH_SCRIPTS="$( echo ${BIDMACH_SCRIPTS} | sed 's+/cygdrive/\([a-z]\)+\1:+' )"

if [[ -z "$BIDMACH_DATA_HOME" ]]; then
  echo '$BIDMACH_DATA_HOME environment variable not set, aborting!' 1>&2
  exit 1
fi

echo "Loading MNIST8M data"
MNIST8M="${BIDMACH_DATA_HOME}/MNIST8M"
mkdir -p ${MNIST8M}/parts_fine
cd ${MNIST8M}

${WGET} http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist8m.bz2

echo "Uncompressing MNIST8M data"

bunzip2 -c mnist8m.bz2 > mnist8m.libsvm

echo "Splitting MNIST8M data"

if [ ! `uname` = "Darwin" ]; then
    split -l 10000 -a 3 -d mnist8m.libsvm parts_fine/part
else
    split -l 10000 -a 3 mnist8m.libsvm parts_fine/part
    j=0
    for i in {a..z}{a..z}{a..z}; do
	jj=`printf "%03d" $j`
	mv parts_fine/part$i parts_fine/part$jj
	j=$((j+1))
	if [ $j -gt 800 ]; then break; fi
    done
fi

cd ${MNIST8M}/parts_fine
${BIDMACH_SCRIPTS}/../bidmach ${BIDMACH_SCRIPTS}/processmnist8m_finesplit.ssc

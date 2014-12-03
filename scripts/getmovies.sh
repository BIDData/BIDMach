#!/bin/bash

BIDMACH_SCRIPTS="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_SCRIPTS=`readlink -f "${BIDMACH_SCRIPTS}"`
  export WGET='wget --no-check-certificate'
else
  while [ -L "${BIDMACH_SCRIPTS}" ]; do
    BIDMACH_SCRIPTS=`readlink "${BIDMACH_SCRIPTS}"`
  done
  export WGET='curl --retry 2 -O'
fi

export BIDMACH_SCRIPTS=`dirname "$BIDMACH_SCRIPTS"`
cd ${BIDMACH_SCRIPTS}
BIDMACH_SCRIPTS=`pwd`
BIDMACH_SCRIPTS="$( echo ${BIDMACH_SCRIPTS} | sed 's+/cygdrive/\([a-z]\)+\1:+' )" 

echo "Loading movielens 10M data"

ML=${BIDMACH_SCRIPTS}/../data/movielens
mkdir -p ${ML}
cd ${ML}

if [ ! -e ml-10m.zip ]; then
    ${WGET} http://files.grouplens.org/datasets/movielens/ml-10m.zip
fi 

unzip ml-10m.zip
cd ml-10M100k
./split_ratings.sh
for i in 1 2 3 4 5 a b; do
    mv r${i}.train r${i}.train.txt
    mv r${i}.test r${i}.test.txt
done
cd ${BIDMACH_SCRIPTS}

bidmach getmovies.ssc

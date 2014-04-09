#!/bin/sh

BIDMACH_SCRIPTS="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_SCRIPTS=`readlink -f "${BIDMACH_SCRIPTS}"`
else 
  BIDMACH_SCRIPTS=`readlink "${BIDMACH_SCRIPTS}"`
fi
export BIDMACH_SCRIPTS=`dirname "$BIDMACH_SCRIPTS"`
BIDMACH_SCRIPTS="$( echo ${BIDMACH_SCRIPTS} | sed s+/cygdrive/c+c:+ )" 

echo "Loading arabic digits data"

UCI=${BIDMACH_SCRIPTS}/../data/uci
cd $UCI

if [ ! -e Train_Arabic_Digit.txt ]; then
    wget --no-check-certificate https://archive.ics.uci.edu/ml/machine-learning-databases/00195/Train_Arabic_Digit.txt
fi 

if [ ! -e "arabic.fmat.lz4" ]; then
    if [ ! -e "arabic.txt" ]; then
        sed 's/^\s*$/0 0 0 0 0 0 0 0 0 0 0 0 0/g' Train_Arabic_Digit.txt > arabic.txt
    fi
    bidmach "BIDMach.DIGITS.preprocess(\"${UCI}/\",\"arabic\")"
fi

if [ -e "arabic.txt" ]; then
    rm arabic.txt
fi
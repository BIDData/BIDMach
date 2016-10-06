#!/bin/bash

BIDMACH_SCRIPTS="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_SCRIPTS=`readlink -f "${BIDMACH_SCRIPTS}"`
  export WGET='wget -c --no-check-certificate'
else
  while [ -L "${BIDMACH_SCRIPTS}" ]; do
    BIDMACH_SCRIPTS=`readlink "${BIDMACH_SCRIPTS}"`
  done
  export WGET='curl -C - --retry 2 -O'
fi

export BIDMACH_SCRIPTS=`dirname "$BIDMACH_SCRIPTS"`
cd ${BIDMACH_SCRIPTS}
BIDMACH_SCRIPTS=`pwd`
BIDMACH_SCRIPTS="$( echo ${BIDMACH_SCRIPTS} | sed 's+/cygdrive/\([a-z]\)+\1:+' )" 

echo "Loading $1 data"

YP="${BIDMACH_SCRIPTS}/../data/YearPrediction"
mkdir -p ${YP}
cd ${YP}

${WGET} http://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip

unzip YearPredictionMSD.txt.zip

cd ${BIDMACH_SCRIPTS}

bidmach getyearprediction.ssc


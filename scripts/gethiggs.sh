#!/bin/bash
set -euo pipefail

BIDMACH_SCRIPTS="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_SCRIPTS=`readlink -f "${BIDMACH_SCRIPTS}"`
  export WGET='wget'
else
  while [ -L "${BIDMACH_SCRIPTS}" ]; do
    BIDMACH_SCRIPTS=`readlink "${BIDMACH_SCRIPTS}"`
  done
  export WGET='curl -C - --retry 2 -O'
fi

export BIDMACH_SCRIPTS=`dirname "$BIDMACH_SCRIPTS"`
cd "${BIDMACH_SCRIPTS}"
BIDMACH_SCRIPTS=`pwd`
BIDMACH_SCRIPTS="$( echo ${BIDMACH_SCRIPTS} | sed 's+/cygdrive/\([a-z]\)+\1:+' )"

UCI_HIGGS="${BIDMACH_SCRIPTS}/../data/uci/Higgs"
mkdir -p "${UCI_HIGGS}"
cd "${UCI_HIGGS}"

if [ ! -e "HIGGS.csv.gz" ]; then
  ${WGET} "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
fi
mkdir -p parts
zcat HIGGS.csv.gz | split -a 3 -d -l 100000 - parts/data

cd "${BIDMACH_SCRIPTS}"
../bidmach higgsprep.ssc

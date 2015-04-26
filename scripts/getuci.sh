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

echo "Loading $1 data"

UCI="${BIDMACH_SCRIPTS}/../data/uci/${1}"
mkdir -p ${UCI}
cd ${UCI}

if [ ! -e docword.txt.gz ]; then
    ${WGET} http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.${1}.txt.gz
    mv docword.${1}.txt.gz docword.txt.gz
fi 
if [ ! -e vocab.txt ]; then
    ${WGET} http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.${1}.txt
    mv vocab.${1}.txt vocab.txt
fi 

echo "Uncompressing docword.${1}.txt.gz"
gunzip -c "docword.txt.gz" | tail -n +4 > "docword.txt"
${BIDMACH_SCRIPTS}/../cbin/tparse.exe -i "docword.txt" -f "${UCI}/../../uci_fmt.txt" -o "" -m "" -d " " -c
${BIDMACH_SCRIPTS}/../cbin/tparse.exe -i "vocab.txt" -f "${UCI}/../../uci_wfmt.txt" -o "" -m "" -c
cd ${BIDMACH_SCRIPTS}/..
cd ${UCI}
${BIDMACH_SCRIPTS}/../bidmach ${BIDMACH_SCRIPTS}/getuci.ssc
mv "smat.lz4" "../${1}.smat.lz4"
mv "term.sbmat.gz" "../${1}.term.sbmat.gz"
mv "term.imat.gz" "../${1}.term.imat.gz"
if [ -e "docword.txt" ]; then
    echo "clearing up"
    rm docword.txt
fi

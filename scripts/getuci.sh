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

UCI="${BIDMACH_SCRIPTS}/../data/uci"
mkdir -p ${UCI}
cd ${UCI}

if [ ! -e docword.${1}.txt.gz ]; then
    ${WGET} http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.${1}.txt.gz
fi 
if [ ! -e vocab.${1}.txt ]; then
    ${WGET} http://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.${1}.txt
fi 

echo "Uncompressing docword.${1}.txt.gz"
gunzip -c "docword.${1}.txt.gz" | tail -n +4 > "docword.${1}.txt"
${BIDMACH_SCRIPTS}/../bin/tparse.exe -i "docword.${1}.txt" -f "${UCI}/../uci_fmt.txt" -o "./${1}." -m "./${1}." -d " " -c
${BIDMACH_SCRIPTS}/../bin/tparse.exe -i "vocab.${1}.txt" -f "${UCI}/../uci_wfmt.txt" -o "./${1}." -m "./${1}." -c
cd ${BIDMACH_SCRIPTS}/..
${BIDMACH_SCRIPTS}/../bidmach "-e" "BIDMach.Experiments.NYTIMES.preprocess(\"${UCI}/\",\"${1}.\")" 
cd ${UCI}
if [ -e "docword.${1}.txt" ]; then
    echo "clearing up"
    rm docword.${1}.txt
    rm vocab.${1}.txt
fi

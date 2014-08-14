#!/bin/bash

BIDMACH_SCRIPTS="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_SCRIPTS=`readlink -f "${BIDMACH_SCRIPTS}"`
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
    curl --retry 2 -O https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.${1}.txt.gz
fi 

if [ ! -e "${1}.smat.lz4" ]; then
    if [ ! -e "docword.${1}.txt" ]; then
        echo "Uncompressing docword.${1}.txt.gz"
        gunzip -c "docword.${1}.txt.gz" | tail -n +4 > "docword.${1}.txt"
    fi
    if [ ! -e "${1}.cols.imat.gz" ]; then
        ${BIDMACH_SCRIPTS}/../bin/tparse.exe -i "docword.${1}.txt" -f "${UCI}/../uci_fmt.txt" -o "./${1}." -m "./${1}." -d " " -c
    fi
    ${BIDMACH_SCRIPTS}/../bidmach "-e"  "BIDMach.NYTIMES.preprocess(\"${UCI}/\",\"${1}.\")"
fi

if [ -e "docword.${1}.txt" ]; then
    echo "clearing up"
#    rm docword.${1}.txt
fi

#!/bin/sh

BIDMACH_SCRIPTS="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_SCRIPTS=`readlink -f "${BIDMACH_SCRIPTS}"`
else 
  BIDMACH_SCRIPTS=`readlink "${BIDMACH_SCRIPTS}"`
fi
BIDMACH_SCRIPTS=`dirname "$BIDMACH_SCRIPTS"`
BIDMACH_SCRIPTS="$( echo ${BIDMACH_SCRIPTS} | sed 's+/cygdrive/\([a-z]\)+\1:+' )" 
echo $BIDMACH_SCRIPTS
echo "Loading RCV1 v2 data"

RCV1=${BIDMACH_SCRIPTS}/../data/rcv1
cd $RCV1

# Get test and training sets
for i in `seq 0 3`; do 
    if [ ! -e lyrl2004_tokens_test_pt${i}.dat.gz ]; then
        wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt${i}.dat.gz
    fi
    if [ $i -eq "0" ]; then
        allfiles=lyrl2004_tokens_test_pt0.dat.gz
    else
        allfiles=${allfiles},lyrl2004_tokens_test_pt${i}.dat.gz
    fi
done
if [ ! -e lyrl2004_tokens_train.dat.gz ]; then
    wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_train.dat.gz
fi
allfiles=${allfiles},lyrl2004_tokens_train.dat.gz

# Get topic assignments
if [ ! -e rcv1-v2.topics.qrels.gz ]; then
    wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz
fi

# Tokenize the text files
if [ ! -e tokenized/dict.gz ]; then
# Windows cant use an uncompression pipe so uncompress here
    if [ "$OS" = "Windows_NT" ]; then
        for i in `seq 0 3`; do 
            if [ ! -e lyrl2004_tokens_test_pt${i}.dat ]; then
                echo "Uncompressing lyrl2004_tokens_test_pt${i}.dat.gz"
                gunzip -c lyrl2004_tokens_test_pt${i}.dat.gz > lyrl2004_tokens_test_pt${i}.dat
            fi
        done
        if [ ! -e lyrl2004_tokens_train.dat ]; then
            gunzip -c lyrl2004_tokens_train.dat.gz > lyrl2004_tokens_train.dat
        fi
        allfiles=`echo $allfiles | sed s/\.dat\.gz/.dat/g`
    fi

    ${BIDMACH_SCRIPTS}/../bin/trec.exe -i $allfiles -o tokenized/ -c
fi

# Parse the topic assignment file
if [ ! -e "catname.imat" ]; then
    ${BIDMACH_SCRIPTS}/../bin/tparse.exe -i rcv1-v2.topics.qrels.gz -f ${RCV1}/fmt.txt -o ./ -m ./ -d " "
fi

# Call bidmach to put the data together
if [ ! -e "docs.smat.lz4" ]; then
    bidmach "BIDMach.RCV1.prepare(\"${RCV1}/tokenized/\")"
fi



#!/bin/bash

SCRIPTDIR=`pwd`

cd ../data/word2vec/raw

if [ ! -e 1-billion-word-language-modeling-benchmark-r13output.tar.gz ]; then
echo "Downloading"
wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
fi

if [ ! -d 1-billion-word-language-modeling-benchmark-r13output ]; then
echo "Uncompressing"
tar xvzf 1-billion-word-language-modeling-benchmark-r13output.tar.gz
# fix the misplaced first news item
mv 1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100 \
   1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled
fi

cd 1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled

FILES=`echo news.en*00100 | sed 's/ /,/g'`

mkdir -p ${SCRIPTDIR}/../data/word2vec/tokenized
mkdir -p ${SCRIPTDIR}/../data/word2vec/tokenized2
mkdir -p ${SCRIPTDIR}/../data/word2vec/data

${SCRIPTDIR}/../cbin/tparse2.exe -i "${FILES}" -f ../../fmt.txt -o ${SCRIPTDIR}/../data/word2vec/tokenized/ -c

cd ${SCRIPTDIR}/../data/word2vec/raw/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/

FILES=`echo news.en*00050 | sed 's/ /,/g'`

${SCRIPTDIR}/../cbin/tparse2.exe -i "${FILES}" -f ../../fmt.txt -o ${SCRIPTDIR}/../data/word2vec/tokenized2/ -c

cd ${SCRIPTDIR}

bidmach getw2vdata.ssc



#!/bin/bash
set +e

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
echo "Downloading Phonetisaurus CMUdict data (more details at https://github.com/cmusphinx/g2p-seq2seq)"

DATADIR=${DATADIR:-${BIDMACH_SCRIPTS}/../data/phonetisaurus-cmudict-split}
mkdir -p ${DATADIR}/raw
mkdir -p ${DATADIR}/json_data
mkdir -p ${DATADIR}/smat_data
pushd ${DATADIR}

pushd raw
${WGET} 'https://downloads.sourceforge.net/project/cmusphinx/G2P%20Models/phonetisaurus-cmudict-split.tar.gz'
tar -xvf phonetisaurus-cmudict-split.tar.gz

popd
pushd json_data
if [ ! -f data_utils.py ]; then
  ${WGET} 'https://raw.githubusercontent.com/cmusphinx/g2p-seq2seq/d6f0f7f0affe896ffd153900e62c02a99fd5af59/g2p_seq2seq/data_utils.py'
  sed -i '/tensorflow/d' data_utils.py
fi
python - <<- EOM
import data_utils
train_gr_ids, train_ph_ids, valid_gr_ids, valid_ph_ids, gr_vocab, ph_vocab, test_dic = \
  data_utils.prepare_g2p_data(None, '../raw/phonetisaurus-cmudict-split/cmudict.dic.train', None, '../raw/phonetisaurus-cmudict-split/cmudict.dic.test')
print('Serializing Python data structures to JSON...')
import json
json.dump(train_gr_ids, open('train.grapheme.json', 'w'))
json.dump(train_ph_ids, open('train.phoneme.json', 'w'))
json.dump(valid_gr_ids, open('valid.grapheme.json', 'w'))
json.dump(valid_ph_ids, open('valid.phoneme.json', 'w'))
json.dump(gr_vocab, open('grapheme.vocab.json', 'w'))
json.dump(ph_vocab, open('phoneme.vocab.json', 'w'))
json.dump(test_dic, open('test_dic_raw.json', 'w'))
print('Done.')
EOM

popd
CMUDICT_JSON_DIR=$PWD/json_data/ CMUDICT_SMAT_DIR=$PWD/smat_data/ ${BIDMACH_SCRIPTS}/../bidmach ${BIDMACH_SCRIPTS}/process_cmudict_json.ssc

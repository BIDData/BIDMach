#!/bin/bash

# This script should be run from the BIDMach/scripts directory

BIDMACH_SCRIPTS="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_SCRIPTS=`readlink -f "${BIDMACH_SCRIPTS}"`
else 
  while [ -L "${BIDMACH_SCRIPTS}" ]; do
    BIDMACH_SCRIPTS=`readlink "${BIDMACH_SCRIPTS}"`
  done
fi
export BIDMACH_SCRIPTS=`dirname "$BIDMACH_SCRIPTS"`
cd ${BIDMACH_SCRIPTS}
BIDMACH_SCRIPTS=`pwd`
BIDMACH_SCRIPTS="$( echo ${BIDMACH_SCRIPTS} | sed 's+/cygdrive/\([a-z]\)+\1:+' )" 


${BIDMACH_SCRIPTS}/getrcv1.sh

${BIDMACH_SCRIPTS}/getuci.sh nips

${BIDMACH_SCRIPTS}/getuci.sh nytimes

# this one is huge, make sure you really want it
# ${BIDMACH_SCRIPTS}/getuci.sh pubmed

${BIDMACH_SCRIPTS}/getdigits.sh

${BIDMACH_SCRIPTS}/getmovies.sh

${BIDMACH_SCRIPTS}/getmnist8m.sh




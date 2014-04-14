#!/bin/sh

BIDMACH_SCRIPTS="${BASH_SOURCE[0]}"
if [ ! `uname` = "Darwin" ]; then
  BIDMACH_SCRIPTS=`readlink -f "${BIDMACH_SCRIPTS}"`
else 
  BIDMACH_SCRIPTS=`readlink "${BIDMACH_SCRIPTS}"`
fi
export BIDMACH_SCRIPTS=`dirname "$BIDMACH_SCRIPTS"`

${BIDMACH_SCRIPTS}/getrcv1.sh

${BIDMACH_SCRIPTS}/getuci.sh nytimes

${BIDMACH_SCRIPTS}/getuci.sh pubmed

${BIDMACH_SCRIPTS}/getuci.sh nips

${BIDMACH_SCRIPTS}/getdigits.sh




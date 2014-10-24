#!/bin/bash

BIDMACH_SCRIPTS="${BASH_SOURCE[0]}"
if [ `uname` = "Darwin" ]; then
    alias wget='curl --retry 2 -O'
else
    BIDMACH_SCRIPTS=`readlink -f "${BIDMACH_SCRIPTS}"`
fi
export BIDMACH_SCRIPTS=`dirname "$BIDMACH_SCRIPTS"`

# ${BIDMACH_SCRIPTS}/getrcv1.sh

${BIDMACH_SCRIPTS}/getuci.sh nips

${BIDMACH_SCRIPTS}/getuci.sh nytimes

# ${BIDMACH_SCRIPTS}/getuci.sh pubmed

${BIDMACH_SCRIPTS}/getdigits.sh




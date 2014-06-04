#!/bin/bash
#set -x
export JAVA_OPTS=-Xmx28G
export LD_LIBRARY_PATH=/home/ubuntu/lib/BIDMat/lib:/usr/local/lib
export PATH=$1
export ALL_LIBS=$2

scala -cp $ALL_LIBS Twitter 41652230 $3 $4 $5 $6 machines



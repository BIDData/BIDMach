#!/bin/bash
#set -x
export JAVA_OPTS=-Xmx60G
export LD_LIBRARY_PATH=/home/ubuntu/lib/BIDMat/lib:/usr/local/lib
export PATH=$1
export ALL_LIBS=$2

scala -cp $ALL_LIBS Yahoo 1413511394 $3 $4 $5 $6 rmachines




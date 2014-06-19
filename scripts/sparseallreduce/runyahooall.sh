#!/bin/bash
#set -x
hosts=`cat $1`
imachine=0
config="4,4,2,2"

for i in `echo $hosts`; do
  host=`echo $i`
  echo $imachine
  ssh -i /home/ubuntu/.ssh/supermario.pem ubuntu@$host "cd /home/ubuntu/sparseallreduce/PageRank;nohup ./runyahoo.sh $PATH $ALL_LIBS $config $imachine 30000000 1 >& /disk4/log-yahoo-$config-$imachine &" &
  imachine=`expr $imachine + 1` 
done



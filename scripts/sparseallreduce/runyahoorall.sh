#!/bin/bash
#set -x
hosts=`cat $1`
imachine=0
config="128"

for i in `echo $hosts`; do
  host=`echo $i`
  echo $imachine
  ssh -i /home/ubuntu/.ssh/supermario.pem ubuntu@$host "cd /home/ubuntu/sparseallreduce/PageRank;nohup ./runyahoor.sh $PATH $ALL_LIBS $config $imachine 30000000 2 >& /disk4/log-yahoor-$config-$imachine &" &
  imachine=`expr $imachine + 1` 
done



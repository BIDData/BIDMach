#!/bin/bash
#set -x
hosts=`cat $1`
imachine=0
config="8,8"

for i in `echo $hosts`; do
  host=`echo $i`
  echo $imachine
  ssh -i /home/ubuntu/.ssh/supermario.pem ubuntu@$host "cd /home/ubuntu/sparseallreduce/PageRank;nohup ./runtwitter.sh $PATH $ALL_LIBS $config $imachine 10000000 1 >& /disk3/log-twitter-$config-$machine &" &
  imachine=`expr $imachine + 1` 
done



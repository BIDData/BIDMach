#!/bin/bash
#set -x
hosts=`cat $1`

for i in `echo $hosts`; do
  host=`echo $i`
  ssh -i /home/ubuntu/.ssh/supermario.pem ubuntu@$host "cd /home/ubuntu/sparseallreduce/PageRank;./kill.sh;" &
done



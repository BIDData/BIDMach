#!/bin/bash
while read host<&4 && read volume<&5
do
  ssh -i /home/ubuntu/.ssh/supermario.pem ubuntu@$host "cd /home/ubuntu/sparseallreduce/PageRank;./mount.sh $volume;" &
done 4<$1 5<$2

#!/bin/sh
hosts=`cat $1`
counter=0;

for i in `echo $hosts`; do
  host=`echo $i`
  ssh -i /home/ubuntu/.ssh/supermario.pem ubuntu@$host "cd /home/ubuntu/sparseallreduce/PageRank/;./check.sh"
  counter=`expr $counter + 1`
  echo $counter
done


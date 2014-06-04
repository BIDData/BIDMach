#!/bin/sh
hosts=`cat $1`
counter=0;

for i in `echo $hosts`; do
  host=`echo $i`
  ssh -i /home/ubuntu/.ssh/supermario.pem ubuntu@$host "cp /disk4/log* /disk4/copylog/" &
  sleep 1s
  scp -i /home/ubuntu/.ssh/supermario.pem ubuntu@$host:/disk4/copylog/log* /logs &
  counter=`expr $counter + 1`
  echo $counter
done


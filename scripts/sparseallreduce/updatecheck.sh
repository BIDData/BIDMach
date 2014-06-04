#!/bin/bash
hosts=`cat $1`

for i in `echo $hosts`; do
  host=`echo $i`
  result=$(ssh -i /home/ubuntu/.ssh/supermario.pem ubuntu@$host "ls machines | wc -l;" 2>&1)
  if [ "$result" -ne 1 ];
  then
    echo "$host: false"
  fi
done

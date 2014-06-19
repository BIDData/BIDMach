#!/bin/sh
hosts=`cat $1`
counter=0;

for i in `echo $hosts`; do
  host=`echo $i`
  ping -c 1 $host
done


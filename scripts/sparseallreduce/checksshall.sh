#!/bin/sh
# $1 machine ips from one placement group $2 number of placement group
hosts=`cat $1`
#remove all existing ips
rm -r ips/$2/*
for i in `echo $hosts`; do
  host=`echo $i`
  ./checkssh.sh $host $2 &
done


#!/bin/bash 

path=$1
number=$2

i=0
while read slave; do
    slaves[$i]=$slave
    i=$((i+1))
done < /code/BIDMach/conf/slaves

alen=$i
echo ${slaves[*]}

j=0
k=0
for i in `seq 0 $number`; do
    fromname=`printf $path $i`
    toname=`printf $path $j`
    echo scp $fromname ${slaves[$k]}:$toname
    scp $fromname ${slaves[$k]}:$toname
    k=$((k+1))
    if [ ${k} -ge ${alen} ]; then
       k=0
       j=$((j+1))
    fi
done

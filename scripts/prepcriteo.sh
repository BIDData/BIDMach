#!/bin/bash

cd d:/criteo

#split -d -l 600000 train.txt parts/train
#split -d -l 600000 test.txt parts/test

cd parts

for i in `seq 0 0`; do
    fnum=`printf "%02d" $i`
    cut -f 1-13 "train$fnum" > "num$fnum"
    cut -f 14-39 "train$fnum" > "hex$fnum"
done
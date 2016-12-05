#!/bin/bash

cd /mnt/BIDMach/data/criteo
mkdir -p parts

split -a 2 -d -l 500000 train.txt parts/train
split -a 2 -d -l 500000 test.txt parts/test

cd /opt/BIDMach/scripts

bidmach readcriteo.ssc

bidmach buildcriteo.ssc

bidmach sortcriteo.ssc

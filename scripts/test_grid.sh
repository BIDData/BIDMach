#!/bin/bash

#read testing parameters from user
read -p 'nodeNum: ' nn
read -p 'dataSize: ' ds
read -p 'threshold: ' th
read -p 'maxRound: ' mr

#set parameters on master 
sed -i "s/^val nodeNum .*$/val nodeNum = ${nn}/" testAllReduceGridMaster.scala

#run testAllReduceGridMaster
nohup bidmach ./testAllReduceGridMaster.scala &

#set parameters on slaves
runall.sh "cd /code/BIDMach/scripts;sed -i \"s/^val maxRound .*$/val maxRound = ${mr}/\" testAllReduceNode.scala;sed -i \"s/^val dataSize.*$/val dataSize = ${ds}/\" testAllReduceNode.scala;sed -i \"s/^val threshold = ThresholdConfig(thAllreduce = .*$/val threshold = ThresholdConfig(thAllreduce = ${th}f, thReduce = ${th}f, thComplete = ${th}f)/\" testAllReduceNode.scala"

#run testAllReduceNode on each slave
./start_workers.sh /code/BIDMach/scripts/testAllReduceNode.scala

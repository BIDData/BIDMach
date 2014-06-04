#!/bin/bash
#set -x
hosts=`cat $1`
imachine=0

for i in `echo $hosts`; do
  host=`echo $i`
  scp -i /home/ubuntu/.ssh/supermario.pem *.sh ubuntu@$host:sparseallreduce/PageRank/ &
  scp -i /home/ubuntu/.ssh/supermario.pem machines ubuntu@$host:sparseallreduce/PageRank/ &
  scp -i /home/ubuntu/.ssh/supermario.pem rmachines ubuntu@$host:sparseallreduce/PageRank/machines &
  #scp -i /home/ubuntu/.ssh/supermario.pem Twitter* ubuntu@$host:sparseallreduce/PageRank/ &
  #scp -i /home/ubuntu/.ssh/supermario.pem Yahoo* ubuntu@$host:sparseallreduce/PageRank/ &
  #scp -i /home/ubuntu/.ssh/supermario.pem ~/lib/BIDMat/BIDMat.jar ubuntu@$host:lib/BIDMat/ &
done



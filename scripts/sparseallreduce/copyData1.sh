#/bin/bash
#set -x
volumes=`cat $1`
counter=0;
#sudo mkdir /disk11
#sudo chown -R ubuntu disk11
#sudo chgrp -R ubuntu disk11

for i in `echo $volumes`; do
  volume=`echo $i`
  hcounter=`expr $counter / 2`
  while [ $(ls /dev | grep xvdl | wc -l) -eq 1 ] || [ $(lsblk | grep xvdl | wc -l) -eq 1 ]; do
    sleep 2s
  done
  ec2-attach-volume $volume -i i-a0f0828e -d /dev/xvdl
  sleep 10s
  while [ $(sudo file -s /dev/xvdl | grep ERROR | wc -l) -eq 1 ]; do
    sleep 1s
  done
  sudo mkfs -t ext4 /dev/xvdl 
  
  if [ ! -d /disk11 ]; then          # dont want errors if it already exists
     sudo mkdir /disk11
  fi
  sudo chown -R ubuntu /disk11      # these lines are probably not necessary, but just in case...
  sudo chgrp -R ubuntu  /disk11
  #sudo mkdir disk
  sudo mount /dev/xvdl /disk11
  sudo chown -R ubuntu /disk11
  sudo chgrp -R ubuntu /disk11
  
  chmod 755 /disk11
  if [ ! -d /disk11/copylog ]; then
    mkdir /disk11/copylog
  fi
  chmod 755 /disk11/copylog

  if [ $counter -lt 64 ];
  then
    #cp disk/inboundIndicesNew-64-$counter.mat /disk11/
    #cp disk/outboundIndicesNew-64-$counter.mat /disk11/
    cp disk/inverts64_p$(printf %03d $counter).imat.lz4 /disk11/
    cp disk/outverts64_p$(printf %03d $counter).imat.lz4 /disk11/
  fi
  #cp disk/inboundIndicesNew-128-$hcounter.mat /disk11/
  #cp disk/outboundIndicesNew-128-$hcounter.mat /disk11/
  #cp disk/inboundIndicesNew-256-$counter.mat /disk11/
  #cp disk/outboundIndicesNew-256-$counter.mat /disk11/
  cp disk/inverts128_p$(printf %03d $hcounter).imat.lz4 /disk11/
  cp disk/outverts128_p$(printf %03d $hcounter).imat.lz4 /disk11/
  cp disk/inverts256_p$(printf %03d $counter).imat.lz4 /disk11/
  cp disk/outverts256_p$(printf %03d $counter).imat.lz4 /disk11/
  sudo umount -d /dev/xvdl
  ec2-detach-volume $volume
  sleep 20s
  counter=`expr $counter + 1`
done



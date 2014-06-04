#/bin/bash
#set -x
volumes=`cat $1`
counter=192;
#sudo mkdir /disk12
#sudo chown -R ubuntu disk12
#sudo chgrp -R ubuntu disk12

for i in `echo $volumes`; do
  volume=`echo $i`
  hcounter=`expr $counter / 2`
  while [ $(ls /dev | grep xvdi | wc -l) -eq 1 ] || [ $(lsblk | grep xvdi | wc -l) -eq 1 ]; do
    sleep 2s
  done
  ec2-attach-volume $volume -i i-60e2934e -d /dev/xvdi
  sleep 10s
  while [ $(sudo file -s /dev/xvdi | grep ERROR | wc -l) -eq 1 ]; do
    sleep 1s
  done
  sudo mkfs -t ext4 /dev/xvdi 
  
  if [ ! -d /disk12 ]; then          # dont want errors if it already exists
     sudo mkdir /disk12
  fi
  sudo chown -R ubuntu /disk12      # these lines are probably not necessary, but just in case...
  sudo chgrp -R ubuntu  /disk12
  #sudo mkdir disk
  sudo mount /dev/xvdi /disk12
  sudo chown -R ubuntu /disk12
  sudo chgrp -R ubuntu /disk12
  
  chmod 755 /disk12
  if [ ! -d /disk12/copylog ]; then
    mkdir /disk12/copylog
  fi
  chmod 755 /disk12/copylog

  if [ $counter -lt 64 ];
  then
    #cp disk/inboundIndicesNew-64-$counter.mat /disk12/
    #cp disk/outboundIndicesNew-64-$counter.mat /disk12/
    cp disk/inverts64_p$(printf %03d $counter).imat.lz4 /disk12/
    cp disk/outverts64_p$(printf %03d $counter).imat.lz4 /disk12/
  fi
  #cp disk/inboundIndicesNew-128-$hcounter.mat /disk12/
  #cp disk/outboundIndicesNew-128-$hcounter.mat /disk12/
  #cp disk/inboundIndicesNew-256-$counter.mat /disk12/
  #cp disk/outboundIndicesNew-256-$counter.mat /disk12/
  cp disk/inverts128_p$(printf %03d $hcounter).imat.lz4 /disk12/
  cp disk/outverts128_p$(printf %03d $hcounter).imat.lz4 /disk12/
  cp disk/inverts256_p$(printf %03d $counter).imat.lz4 /disk12/
  cp disk/outverts256_p$(printf %03d $counter).imat.lz4 /disk12/
  sudo umount -d /dev/xvdi
  ec2-detach-volume $volume
  sleep 20s
  counter=`expr $counter + 1`
done



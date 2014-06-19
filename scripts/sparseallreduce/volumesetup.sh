#!/bin/bash

for i in {0..191}
do
  ec2-create-volume --size 1 --availability-zone us-east-1a
done

ec2-describe-volumes --filter "size=1" | grep -o 'vol[a-zA-Z0-9-]\+' > volumesbackup



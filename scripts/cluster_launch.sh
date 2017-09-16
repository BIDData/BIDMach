#!/bin/bash

if [[ "$CLUSTER" == "" ]]; then
    CLUSTER="bidcluster1"
fi

# launch a cluster
python bidmach_ec2.py -k "dss2_rsa" -i ~/.ssh/dss2_rsa -a "ami-ba7281c2" -s 4 --instance-type=p2.xlarge --region=us-west-2 --zone=us-west-2a --vpc-id="vpc-c93fbdac" --subnet-id="subnet-75177210" launch $CLUSTER

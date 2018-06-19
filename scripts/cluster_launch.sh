#!/bin/bash

if [[ "$CLUSTER" == "" ]]; then
    CLUSTER="bidcluster2"
fi

# launch a cluster
python bidmach_ec2.py -k "dss2_rsa" -i ~/.ssh/dss2_rsa -a "ami-b2bf04ca" -s 16 --instance-type=p2.xlarge --region=us-west-2 --zone=us-west-2a --vpc-id="vpc-c93fbdac" --subnet-id="subnet-75177210" --additional-tags='Group:DSS 2' launch $CLUSTER

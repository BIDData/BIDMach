#!/bin/bash

if [[ "$CLUSTER" == "" ]]; then
    CLUSTER="bidcluster1"
fi

# start the cluster
python bidmach_ec2.py -k "dss2_rsa" -i ~/.ssh/dss2_rsa --region=us-west-2 start $CLUSTER


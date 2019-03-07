#!/bin/bash

if [[ "$CLUSTER" == "" ]]; then
    CLUSTER="bidcluster1"
fi

if [ ! ${1} == "" ]; then
    LOGIN="-n ${1} login"
else
    LOGIN="login"
fi

# login to the master
python bidmach_ec2.py -k "dss2_rsa" -i ~/.ssh/dss2_rsa --region=us-west-2 ${LOGIN} $CLUSTER




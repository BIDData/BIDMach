#!/bin/bash
#set -x
status=$(ssh $1 echo ok 2>&1)
if [[ $status == ok ]] ; then
    mkdir ips/$2/$1
fi


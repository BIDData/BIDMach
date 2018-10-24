#!/bin/bash

while read slave; do
    echo ssh "${slave}" "nohup \"${1}\" > ${HOME}/logs/bklog.txt 2>&1 &"
    ssh -n -o StrictHostKeyChecking=no "${slave}" "nohup sh -c \"${1}\" > ${HOME}/logs/bklog.txt 2>&1 &"
done < /code/BIDMach/conf/slaves

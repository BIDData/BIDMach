#!/bin/bash
folder=`dirname ${1}`
while read slave; do
    echo "distributing to ${slave}"
    rsync "${1}" "${slave}:${folder}"
done < /code/BIDMach/conf/slaves

#!/bin/bash
set -e

while read slave; do
    echo "distributing to ${slave}"
    rsync -r "${1}/" "${slave}:${1}"
done < /code/BIDMach/conf/slaves

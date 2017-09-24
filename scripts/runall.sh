#!/bin/bash

while read slave; do
    echo ssh "${slave}" "${1}"
    ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${slave}" "${1}"
done < /code/BIDMach/conf/slaves

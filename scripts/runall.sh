#!/bin/bash

while read slave; do
    echo ssh "${slave}" "${1}"
    ssh -n -o StrictHostKeyChecking=no "${slave}" "${1}"
done < /code/BIDMach/conf/slaves

#!/bin/bash

echo "Starting the Grid Master script here"
screen -d -m bash -i -x /code/BIDMach/scripts/runmaster16.sh

echo "Waiting 20 seconds for Master startup"
sleep 20

echo "Starting Nodes"
runall.sh 'screen -d -m bash -i -x /code/BIDMach/scripts/runnode16.sh'

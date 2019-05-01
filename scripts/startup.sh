#!/bin/bash

echo "Starting the Grid Master script here"
screen -d -m bash -i -x /code/BIDMach/scripts/runmaster.sh

echo "Waiting 20 seconds for Master startup"
sleep 20

echo "Starting Nodes"
runall.sh 'screen -d -m bash -i -x /code/BIDMach/scripts/runnode.sh'

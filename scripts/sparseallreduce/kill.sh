#!/bin/bash
ps aux | grep mount | awk '{print $2}' | xargs sudo kill 9
ps aux | grep scala | awk '{print $2}' | xargs sudo kill 15
sleep 3s
ps aux | grep scala | awk '{print $2}' | xargs sudo kill 2
sleep 3s
ps aux | grep scala | awk '{print $2}' | xargs sudo kill 9

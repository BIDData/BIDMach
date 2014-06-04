#/bin/bash

if [ ! -s /home/ubuntu/sparseallreduce/PageRank/done.mount ]; then
  echo "mount fail"
fi
if [ ! -s /home/ubuntu/sparseallreduce/PageRank/machines ]; then
  echo "missing machines"
fi
if [ ! -s /home/ubuntu/sparseallreduce/PageRank/rmachines ]; then
  echo "missing rmachines"
fi

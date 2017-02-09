#!/bin/bash
set -e

if [[ -z $1 ]]; then
  echo 'Must supply script argument for workers to start!' 1>&2
  echo 'Example: ./start_workers.sh distributed/worker_criteo_lr.ssc' 1>&2
  exit 1
fi

WORKER_SCRIPT="${PWD}/${1}"

SSH_OPTS='-T -o ConnectTimeout=3'
while read worker_ip; do
    echo "Starting BIDMach worker on ${worker_ip}"
    ssh $SSH_OPTS "ec2-user@${worker_ip}" << EOS

    JAVA_OPTS=$JAVA_OPTS nohup bidmach $WORKER_SCRIPT </dev/null >/tmp/bidmach_worker.log 2>&1 & disown

EOS
done < /opt/spark/conf/slaves
echo 'Done!'

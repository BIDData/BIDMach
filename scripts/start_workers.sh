#!/bin/bash
set -e
SSH_OPTS='-T -o ConnectTimeout=3'
while read worker_ip; do
    echo "Starting BIDMach worker on ${worker_ip}"
    ssh $SSH_OPTS "ec2-user@${worker_ip}" << EOS

    nohup bidmach /opt/BIDMach/scripts/testrecv.ssc </dev/null >/tmp/bidmach_worker.log 2>&1 &

EOS
done < /opt/spark/conf/slaves
echo 'Done!'

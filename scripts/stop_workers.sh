#!/bin/bash
set -e
SSH_OPTS='-T -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=no'
while read worker_ip; do
    echo "Killing BIDMach worker on ${worker_ip}"
    ssh $SSH_OPTS "ec2-user@${worker_ip}" << EOS

    jps | grep 'MainGenericRunner' | awk '{print \$1}' | xargs -I% kill %

EOS
done < /opt/spark/conf/slaves
echo 'Done!'

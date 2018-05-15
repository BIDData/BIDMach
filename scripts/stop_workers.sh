#!/bin/bash
set -e
SSH_OPTS='-T -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=no'
while read worker_ip; do
    echo "Killing BIDMach worker on ${worker_ip}"
    ssh $SSH_OPTS "ubuntu@${worker_ip}" << EOS

    jps | grep 'MainGenericRunner' | awk '{print \$1}' | xargs -I% kill %

EOS
done < /code/BIDMach/conf/slaves
echo 'Done!'

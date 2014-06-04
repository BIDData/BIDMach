#/bin/bash
export JAVA_HOME="/usr/"
export EC2_HOME="/home/ubuntu/lib/ec2-api-tools-1.6.7.2"
export PATH=$PATH:$EC2_HOME/bin
export AWS_ACCESS_KEY=AKIAI5K535QYN4P6OSJA
export AWS_SECRET_KEY=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

sudo umount -d /dev/xvdk
ec2-detach-volume $1

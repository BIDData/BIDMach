#/bin/bash
export JAVA_HOME="/usr/"
export EC2_HOME="/home/ubuntu/lib/ec2-api-tools-1.6.7.2"
export PATH=$PATH:$EC2_HOME/bin
export AWS_ACCESS_KEY=AAAA
export AWS_SECRET_KEY=BBBB

sudo umount -d /dev/xvdk
ec2-detach-volume $1

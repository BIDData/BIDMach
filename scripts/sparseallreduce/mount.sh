#/bin/bash
export JAVA_HOME="/usr/"
export EC2_HOME="/home/ubuntu/lib/ec2-api-tools-1.6.7.2"
export PATH=$PATH:$EC2_HOME/bin
export AWS_ACCESS_KEY=AKIAI5K535QYN4P6OSJA
export AWS_SECRET_KEY=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

if [ ! -d /disk4 ]; then
  sudo mkdir /disk4
fi
#dir for copy logs
if [ ! -d /disk4/copylog ]; then
  sudo mkdir /disk4/copylog
fi
sudo chown -R ubuntu /disk4
sudo chgrp -R ubuntu /disk4
sudo chmod -R 755 /disk4

ec2-attach-volume $1 -i $(ec2metadata --instance-id) -d /dev/xvdk
sleep 15s
while [ $(sudo file -s /dev/xvdk | grep ERROR | wc -l) -eq 1 ]; do
  sleep 1s
done
sudo mount /dev/xvdk /disk4
sudo chown -R ubuntu /disk4
sudo chgrp -R ubuntu /disk4
sudo chmod -R 755 /disk4

echo "x" > done.mount

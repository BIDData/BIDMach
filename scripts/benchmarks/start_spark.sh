
cd /opt/spark/ec2

# launch a cluster
./spark-ec2 -k "pils_rsa" -i /home/ec2-user/.ssh/jfc_rsa -s 2 --instance-type=m3.xlarge --region=us-west-1 launch sparkcluster

# ganglia patch

MASTER=`./spark-ec2 -k "pils_rsa" -i /home/ec2-user/.ssh/jfc_rsa --region=us-west-1 get-master sparkcluster | tail -n 1`
scp -i ~/.ssh/jfc_rsa ~/httpd.conf ec2-user@${MASTER}:httpd.conf

# login to the master
./spark-ec2 -k "pils_rsa" -i /home/ec2-user/.ssh/jfc_rsa --region=us-west-1 login sparkcluster

rm -r /var/lib/ganglia/rrds
ln -s /mnt/ganglia/rrds /var/lib/ganglia/rrds

# ganglia patch
cp /etc/httpd/conf/httpd.conf /etc/httpd/conf/httpd_bkup.conf
cp /home/ec2-user/httpd.conf /etc/httpd/conf/httpd.conf
apachectl -k graceful

spark/bin/spark-shell

./spark-ec2 -k "pils_rsa" -i /home/ec2-user/.ssh/jfc_rsa --region=us-west-1 destroy sparkcluster



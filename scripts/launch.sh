#!/bin/bash

cd /code/BIDMach/scripts

# launch a cluster
python bidmach_ec2.py -k "dss2_rsa" -i ~/.ssh/dss2_rsa -a "ami-ba7281c2" -s 4 --instance-type=p2.xlarge --region=us-west-2 --zone=us-west-2a --vpc-id="vpc-c93fbdac" --subnet-id="subnet-75177210" launch bidcluster1

# start the cluster
python bidmach_ec2.py -k "dss2_rsa" -i ~/.ssh/dss2_rsa --region=us-west-2 start bidcluster4

# login to the master
python bidmach_ec2.py -k "dss2_rsa" -i ~/.ssh/dss2_rsa --region=us-west-2 login bidcluster4

# stop the cluster
python bidmach_ec2.py -k "dss2_rsa" -i ~/.ssh/dss2_rsa --region=us-west-2 stop bidcluster4

# need more driver memory for several models, e.g. multiclass and word2vec
spark/bin/spark-shell 
/opt/spark/bin/spark-shell --driver-memory=16g --conf "spark.driver.maxResultSize=8g"


echo "y" | python bidmach_ec2.py -k "dss2_rsa" -i ~/.ssh/dss2_rsa --region=us-west-2 destroy bidcluster1



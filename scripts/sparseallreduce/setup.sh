
N=69
TYPE="cc2.8xlarge"
PLACEMENT="sparseallreduce"

ec2-run-instances ami-53c5c03a -n $N -g template-all-access -k supermario -t $TYPE --placement-group $PLACEMENT --availability-zone us-east-1a

sleep 10s

ec2-describe-instances --filter "instance-type=$TYPE" --filter "placement-group-name=$PLACEMENT" | grep -o 'ip[0-9-]\+' > /home/ubuntu/sparseallreduce/PageRank/rawmachines


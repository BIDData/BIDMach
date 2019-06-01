#!/usr/bin/env python

import subprocess
import sys
import os
import datetime

def main():
    files = sys.argv[1:]
    s = subprocess.check_output("python bidmach_ec2.py -k id_rsa -i ~/.ssh/id_rsa --region=us-west-2 get-slaves " + os.environ['CLUSTER'], shell=True)
    slaves = s.splitlines()[2:]
    dir = '/code/BIDMach/%s/%s' % (os.environ['CLUSTER'], datetime.datetime.now().strftime("%Y%m%d%H%M"))
    os.mkdir(dir)
    for s in slaves:
        slave_dir = '%s/%s' % (dir, s)
        os.mkdir(slave_dir)
        todostr = 'rsync -e "ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no" -avz ubuntu@%s:/code/BIDMach/logs/log.0.0.txt %s/' % (s, slave_dir)
        print(todostr)
        subprocess.check_call(todostr, shell=True)
        todostr = 'rsync -e "ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no" -avz ubuntu@%s:/code/BIDMach/scripts/logres* %s/' % (s, slave_dir)
        print(todostr)
        subprocess.check_call(todostr, shell=True)


if __name__ == "__main__":
    main()

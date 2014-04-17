#!/bin/sh

time /code/spark/bin/run-example org.apache.spark.mllib.classification.LogisticRegressionWithSGD local /big/RCV1/v2/spark_train.dat 10000 10 > /code/spark/model.txt

sed 's/Weights:.\[//g' /code/spark/model.txt | sed '/\]$/N;s/\]\nIntercept: /,/'  > /code/spark/modelx.txt

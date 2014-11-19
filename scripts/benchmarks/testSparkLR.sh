#!/bin/sh

SPARKDIR=/code/spark-1.1.0

time ${SPARKDIR}/bin/run-example mllib.classification.LogisticRegressionWithSGD local /big/RCV1/v2/spark_train.dat 10000 10 > ${SPARKDIR}/model.txt

sed 's/Weights:.\[//g' ${SPARKDIR}/model.txt | sed '/\]$/N;s/\]\nIntercept: /,/'  > ${SPARKDIR}/modelx.txt

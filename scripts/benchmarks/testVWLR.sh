#!/bin/sh

time /code/vowpal_wabbit/vowpalwabbit/vw --oaa 110 -f rcv1.model --loss_function logistic -b 24 --adaptive --invariant -l 1 --cache_file vw.cache --passes 1 -d /big/RCV1/v2/vw_sparse_train.dat



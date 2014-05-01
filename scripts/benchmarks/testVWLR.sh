#!/bin/sh

time /code/vowpal_wabbit/vowpalwabbit/vw --oaa 110 -f /code/vowpal_wabbit/rcv1.model --loss_function logistic -b 24 --adaptive --invariant -l 1 --cache_file /code/vowpal_wabbit/rcv1_sparse_cache.dat --passes 1 -d /big/RCV1/v2/vw_sparse_train.dat



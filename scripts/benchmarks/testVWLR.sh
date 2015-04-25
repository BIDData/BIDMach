#!/bin/sh

time /code/vowpal_wabbit/vowpalwabbit/vw --oaa 103 --readable_model rcv1.model.txt --loss_function logistic -b 24 --adaptive --invariant -l 1 --cache_file vw.cache --passes 3 -d ../../data/rcv1/vwtrain.dat



#!/bin/sh

time /code/vowpal_wabbit/vowpalwabbit/vw --lda 256 --lda_D 100000 --passes 3 --readable_model wordTopics.dat --bit_precision 18 --learning_rate 1.0 --lda_rho 0.1 --cache_file vw.cache --data /big/RCV1/v2/vw_sparse_lda_train.dat --lda_alpha 0.1 --random_weights true --power_t 0.5 --minibatch 1024 --initial_t 1.0 

# BIDMach options
#   opts.putBack = 1
#   opts.uiter = 1
#   opts.batchSize = 1024
#   opts.npasses = 3
#   opts.dim = 256
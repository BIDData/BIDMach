import time

import numpy as np
from scipy import sparse
import scipy.io as sio
#import pylab as pl

from sklearn.linear_model import SGDClassifier
from sklearn.datasets.samples_generator import make_blobs

XY=sio.loadmat("/data/rcv1/all.mat")
X=XY["data"]
Y=XY["cats"]
print("Finished reading")
batch_size = 10
dim=256
sgd_means = SGDClassifier(loss='log', alpha=0.01, fit_intercept=true, max_iter=3)
t0 = time.time()
sgd.fit(X,Y)
t_batch = time.time() - t0
print(t_batch)

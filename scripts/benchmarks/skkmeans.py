import time

import numpy as np
import scipy.io as sio
import h5py
from sklearn.datasets import load_svmlight_file
from sklearn.cluster import KMeans

f = h5py.File('/code/BIDMach/data/MNIST8M/all.mat','r')

t0 = time.time()
data = f.get('/all') # Get a certain dataset
X = np.array(data)
t1 = time.time()

t_read = t1 - t0
print("Finished reading in " + repr(t_read) + " secs")

batch_size = 10
kmeans = KMeans(n_clusters=256, init='random', n_init=1, max_iter=10, tol=0.0001, precompute_distances=False, verbose=0, random_state=None, copy_x=False, n_jobs=1)
kmeans.fit(X)
t2 = time.time()
t_batch = t2 - t1
print("compute time " + repr(t_batch) + " secs")

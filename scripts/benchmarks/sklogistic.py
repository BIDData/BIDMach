import time

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_svmlight_file

t0 = time.time()
print("Start reading")
X, Y = load_svmlight_file("../../data/rcv1/train.libsvm")

print("Finished reading")
batch_size = 10

sgd = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.01, fit_intercept=True, n_iter=3))
t1 = time.time()
sgd.fit(X,Y)
t2 = time.time()

print("load time {0:3.2f}, train time {1:3.2f}".format(t1-t0,t2-t1))


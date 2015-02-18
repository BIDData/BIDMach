import time

import numpy as np
from scipy import sparse
import scipy.io as sio
#import pylab as pl

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split

print("Start reading")
XY=sio.loadmat("/code/BIDMach/data/rcv1/all2.mat")
X=XY["data"].transpose()
Y=XY["cats"].transpose()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=0)
Y6=Y_train[:,6]
print("Finished reading")
batch_size = 10
t0 = time.time()
sgd = OneVsRestClassifier(SGDClassifier(loss='log', verbose=0, alpha=1.0e-6, penalty='l1', n_jobs=1, n_iter=1))
sgd.fit(X_train,Y_train)

t1 = time.time()
sgd2=SGDClassifier(loss='log', verbose=0, alpha=0.01, fit_intercept=True, n_iter=1)
sgd2.fit(X_train,Y6)
t2 = time.time()
#sgd=LogisticRegression(fit_intercept=True)
#sgd.fit(X,Y6)

t_batch = t1 - t0
Y_score = sgd.decision_function(X_test)
fpr = dict()
tpr = dict()
roc_auc = np.zeros(100)
for i in range(100):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

Y6_score = sgd2.decision_function(X_test)
fpr6, tpr6, _ = roc_curve(Y_test[:,6], Y6_score)
auc6 =auc(fpr6,tpr6)


print(t_batch)
print(t2-t1)

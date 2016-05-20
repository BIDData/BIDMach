import time
import numpy as np;
import numpy.random as rand;
t0 = time.time()
n = 10000
a = rand.rand(n,n)
b = rand.rand(n,n)

t0 = time.time()
c = a + b;
t1 = time.time()
dt1 = t1 - t0
print dt1

for i in range(0,n):
    for j in range(0,n):
        c[i][j] = a[i][j] + b[i][j];

t2 = time.time()

dt2 = t2 - t1
print dt1, dt2

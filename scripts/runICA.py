'''
A testing suite for ICA. This will run some Python code to build the data, then calls the ICA
testing script that contains BIDMach commands, then comes back to this Python code to plot the data.
This code should be in the BIDMach/scripts folder.

(c) February 2015 by Daniel Seita
'''

import matplotlib.pyplot as plt
import numpy as np
import pylab
import sys
from scipy import signal
from sklearn.decomposition import FastICA,PCA
from subprocess import call

'''
Returns a matrix where each row corresponds to one signal. Each row has been standardized to have
zero mean and unit variance (I think), and they also have additive Gaussian noise. In order to
ensure that we actually see enough variation in a small time stamp, the "first" and "second"
variables (and possibly others) are used to increase/decrease the "density" of the data. For
instance a high "first" value pushes the sine waves close together.

> group is an integer that represents the group selection, useful for running many tests
> time is from numpy and controls the density of the data
> num_samples is the number of total samples to use for each row
'''
def get_source(group, time, num_samples):
    S = None
    first = max(2, int(num_samples/4000))
    second = max(3, int(num_samples/3000))
    third = max(2, first/10)
    if group == 1:
        s1 = np.sin(first * time)
        s2 = np.sign(np.sin(second * time))
        s3 = signal.sawtooth(first * np.pi * time)
        S = np.c_[s1, s2, s3]
    elif group == 2:
        s1 = np.sin(first * time)
        s2 = np.sign(np.sin(second * time))
        s3 = signal.sawtooth(first * np.pi * time)
        s4 = signal.sweep_poly(third * time, [1,2])
        S = np.c_[s1, s2, s3, s4]
    elif group == 3:
        s1 = np.cos(second * time)                  # Signal 1: cosineusoidal signal
        s2 = np.sign(np.sin(second * time))         # Signal 2: square signal
        s3 = signal.sawtooth(first * np.pi * time)  # Signal 3: saw tooth signal
        s4 = signal.sweep_poly(third * time, [1,2]) # Signal 4: sweeping polynomial signal
        s5 = np.sin(first * time)                   # Signal 5: sinusoidal signal
        S = np.c_[s1, s2, s3, s4, s5]
    elif group == 4:
        s1 = np.sin(first * time)
        s2 = signal.sawtooth(float(first/2.55) * np.pi * time)
        s3 = np.sign(np.sin(second * time))
        s4 = signal.sawtooth(first * np.pi * time)
        s5 = signal.sweep_poly(third * time, [1,2])
        S = np.c_[s1, s2, s3, s4, s5]
    S += 0.2 * np.random.normal(size=S.shape)
    S /= S.std(axis=0)
    return S.T

'''
Generates mixed data. Note that if whitened = True, this picks a pre-whitened matrix to analyze...
Takes in the group number and returns a mixing matrix of the appropriate size. If the data needs to
be pre-whitened, then we should pick an orthogonal mixing matrix. There are three orthogonal
matrices and three non-orthogonal matrices.
'''
def get_mixing_matrix(group, pre_whitened):
    A = None
    if group == 1:
        if pre_whitened:
            A = np.array([[ 0,  -.8, -.6],
                          [.8, -.36, .48],
                          [.6,  .48, -.64]])
        else:
            A = np.array([[  1, 1, 1],
                          [0.5, 2, 1],
                          [1.5, 1, 2]])
    elif group == 2:
        if pre_whitened:
            A = np.array([[-0.040037,  0.24263, -0.015820,   0.96916],
                          [ -0.54019,  0.29635,   0.78318, -0.083724],
                          [  0.84003,  0.23492,   0.48878, -0.016133],
                          [ 0.030827, -0.89337,   0.38403,   0.23120]])
        else:
            A = np.array([[ 1,    2,  -1, 2.5],
                          [-.1, -.1,   3, -.9],
                          [8,     1,   7,   1],
                          [1.5,  -2,   3,  -1]])
    elif group == 3 or group == 4:
        if pre_whitened:
            A = np.array([[ 0.31571,  0.45390, -0.59557,  0.12972,  0.56837],
                          [-0.32657,  0.47508,  0.43818, -0.56815,  0.39129],
                          [ 0.82671,  0.11176,  0.54879,  0.05170,  0.01480],
                          [-0.12123, -0.56812,  0.25204,  0.28505,  0.71969],
                          [-0.30915,  0.48299,  0.29782,  0.75955, -0.07568]])
        else:
            A = np.array([[ 1,    2,  -1, 2.5,   1],
                          [-.1, -.1,   3, -.9,   2],
                          [8,     1,   7,   1,   3],
                          [1.5,  -2,   3,  -1,   4],
                          [-.1,   4, -.1,   3, -.2]])
    return A

'''
Takes in the predicted source from BIDMach and the original source and attempts to change the order
and cardinality of the predicted data to match the original one. This is purely for debugging. newS
is the list of lists that forms the numpy array, and rows_B_taken ensures a 1-1 correspondence.

> B is the predicted source from BIDMach
> S is the actual source before mixing
'''
def rearrange_data(B, S):
    newS = []
    rows_B_taken = []
    for i in range(S.shape[0]):
        new_row = S[i,:]
        change_sign = False
        best_norm = 99999999
        best_row_index = -1
        for j in range(B.shape[0]):
            if j in rows_B_taken:
                continue
            old_row = B[j,:]
            norm1 = np.linalg.norm(old_row + new_row)
            if norm1 < best_norm:
                best_norm = norm1
                best_row_index = j
                change_sign = True
            norm2 = np.linalg.norm(old_row - new_row)
            if norm2 < best_norm:
                best_norm = norm2
                best_row_index = j
        rows_B_taken.append(best_row_index)
        if change_sign:
            newS.append((-B[best_row_index,:]).tolist())
        else:
            newS.append(B[best_row_index,:].tolist())
    return np.array(newS)


########
# MAIN #
########

# Some administrative stuff to make it clear how to handle this code.
if len(sys.argv) != 5:
    print "\nUsage: python runICA.py <num_samples> <data_group> <pre_zero_mean> <pre_whitened>"
    print "<num_samples> should be an integer; recommended to be at least 10000"
    print "<data_group> should be an integer; currently only {1,2,3,4} are supported"
    print "<pre_zero_mean> should be \'Y\' or \'y\' if you want the (mixed) data to have zero-mean"
    print "<pre_whitened> should be \'Y\' or \'y\' if you want the (mixed) data to be pre-whitened"
    print "You also need to call this code in the directory where you can call \'./bidmach scripts/ica_test.ssc\'\n"
    sys.exit()
n_samples = int(sys.argv[1])
data_group = int(sys.argv[2])
pre_zero_mean = True if sys.argv[3].lower() == "y" else False
pre_whitened = True if sys.argv[4].lower() == "y" else False
if data_group < 1 or data_group > 4:
    raise Exception("Data group = " + str(data_group) + " is out of range.")
plot_extra_info = False # If true, plot the mixed input data (X) in addition to the real/predicted sources

# With parameters in pace, generate source, mixing, and output matrices, and save them to files.
np.random.seed(0)
time = np.linspace(0, 8, n_samples) # These need to depend on num of samples
S = get_source(data_group, time, n_samples)
A = get_mixing_matrix(data_group, pre_whitened)
X = np.dot(A,S)
print "\nMean for the mixed data:"
for i in range(X.shape[0]):
    print "Row {}: {}".format(i+1, np.mean(X[i,:]))
print "\nThe covariance matrix for the mixed data is\n{}.".format(np.cov(X))
np.savetxt("ica_source.txt", S, delimiter=" ")
np.savetxt("ica_mixing.txt", A, delimiter=" ")
np.savetxt("ica_output.txt", X, delimiter=" ")
print "\nNow calling ICA in BIDMach...\n"

# Call BIDMach. Note that this will exit automatically with sys.exit, without user intervention.
call(["./bidmach", "scripts/ica_test.ssc"])
print "\nFinished with BIDMach. Now let us plot the data."

# Done with BIDMach. First, for the sake of readability, get distributions in same order.
B = pylab.loadtxt('ica_pred_source.txt')
newB = rearrange_data(B, S)

# Extract data and plot results. Add more colors if needed but 5 is plenty.
plt.figure()
if plot_extra_info:
    models = [X.T, S.T, newB.T]
    names = ['Input to ICA','True Sources Before Mixing','BIDMach\'s FastICA']
else:
    models = [S.T, newB.T]
    names = ['True Sources Before Mixing','BIDMach\'s FastICA']
colors = ['darkcyan', 'red', 'blue', 'orange', 'yellow']
plot_xlim = min(n_samples-1, 10000)
for ii, (model, name) in enumerate(zip(models, names), 1):
    if plot_extra_info:
        plt.subplot(3, 1, ii)
    else:
        plt.subplot(2, 1, ii)
    plt.title(name)
    plt.xlim([0,plot_xlim])
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)
plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()


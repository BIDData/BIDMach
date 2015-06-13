# I'll be using this code to generate some data for a simple Bayesian network.
# (c) 2015 by Daniel Seita

import numpy as np

ncols = 1000000 # Change as needed
nrows = 5
data = np.zeros([nrows,ncols])
# First, handle variables X_0 (intelligence) and X_1 (difficulty)
data[0,:] = np.random.choice(2, ncols, p = [0.7, 0.3])
data[1,:] = np.random.choice(2, ncols, p = [0.6, 0.4])
third = []
fourth = []
fifth = []
for i in range(ncols):
    # Variable X_2 (SAT)
    if data[0,i] == 0:
        third.append( np.random.choice(2, 1, p = [0.95, 0.05])[0] )
    else:
        third.append( np.random.choice(2, 1, p = [0.2, 0.8])[0] )
    # Variable X_3 (grade)
    if (data[0,i] == 0 and data[1,i] == 0):
        fourth.append( np.random.choice(3, 1, p = [0.3, 0.4, 0.3])[0] )
    elif (data[0,i] == 0 and data[1,i] == 1):
        fourth.append( np.random.choice(3, 1, p = [0.05, 0.25, 0.7])[0] )
    elif (data[0,i] == 1 and data[1,i] == 0):
        fourth.append( np.random.choice(3, 1, p = [0.9, 0.08, 0.02])[0] )
    else:
        fourth.append( np.random.choice(3, 1, p = [0.5, 0.3, 0.2])[0] )
    # Variable X_4 (letter)
    if fourth[i] == 0:
        fifth.append( np.random.choice(2, 1, p = [0.1, 0.9])[0] )
    elif fourth[i] == 1:
        fifth.append( np.random.choice(2, 1, p = [0.4, 0.6])[0] )
    else:
        fifth.append( np.random.choice(2, 1, p = [0.99, 0.01])[0] )
data[2,:] = third
data[3,:] = fourth
data[4,:] = fifth
np.savetxt('dataStudent_' + str(ncols) + '.txt', data, fmt='%i')    

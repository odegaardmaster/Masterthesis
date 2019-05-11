# This program is used to create a samplespace with +- 20 % from the standard values


import pyDOE
import csv
from scipy.stats.distributions import uniform
import numpy as np
import time


# Default values
#Cm ,gL ,gNa ,EL ,eNa ,EK ,gc ,gkdr ,gAHP ,gC ,gCa ,ECa ,p
default = np.array([ 3.0e+00,  1.0e-04,  3.0e-02, -6.0e+01,  5.0e+01, -7.5e+01,
        2.1e+00,  1.5e-02,  8.0e-04,  1.5e-02,  1.0e-02,  8.0e+01,
        5.0e-01])





# Creating range used in LHS. ["Start value","Range"] OBS for "negative values, order is changed"
sample_range = np.array([[default[0]*0.8, default[0]*1.2- default[0]*0.8],
                [default[1]*0.8, default[1]*1.2- default[1]*0.8],
                [default[2]*0.8, default[2]*1.2- default[2]*0.8],
                [default[3]*1.2, default[3]*0.8- default[3]*1.2],
                [default[4]*0.8, default[4]*1.2- default[4]*0.8],
                [default[5]*1.2, default[5]*0.8- default[5]*1.2],
                [default[6]*0.8, default[6]*1.2- default[6]*0.8],
                [default[7]*0.8, default[7]*1.2- default[7]*0.8],
                [default[8]*0.8, default[8]*1.2- default[8]*0.8],
                [default[9]*0.8, default[9]*1.2- default[9]*0.8],
                [default[10]*0.8, default[10]*1.2- default[10]*0.8],
                [default[11]*0.8, default[11]*1.2- default[11]*0.8],
                [default[12]*0.8, default[12]*1.2- default[12]*0.8]])


# Define uniform distribution and LHS sampler
ud = uniform(sample_range[:,0],sample_range[:,1]) #Create a uniform distribution
lh = pyDOE.lhs(13,samples=3**13) # Initiation a LHS class.

# Creating samples
samples = ud.ppf(lh) #Restuling in 3**13 samples in a LHS design

t0 = time.time() # Measureing writing time
np.savetxt("savepath",samples, delimiter=",")
print(time.time()-t0) # Measureing writing time





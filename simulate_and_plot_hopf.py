#%% Import necessary packages
import numpy as np
import matplotlib.pyplot as pp
from scipy.io import loadmat

#%% Load some data and set some parameters
path = '../git_introduction/'
freq = np.load(path + 'hopf_frequencies.npy', allow_pickle = True)
SC   = np.load(path + 'SC.npy', allow_pickle = True)
pars = {'n_runs':1, 'n_roi':90, 'TR':0.1, 'dt':0.025, 'n_time': 50}
tss = hf.sim_hopf(freq, SC, G = 3.0, **pars)

#%% Plot figure
pp.figure()
pp.xlabel('timesteps')
pp.title('Example timeseries Hopf model')
pp.plot(tss[0,:,:].T, linewidth = 0.2)
pp.show()

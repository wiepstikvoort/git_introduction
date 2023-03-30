
#%% Import necessary packages
import numpy as np
from numpy import matlib
from scipy import signal

#%% Function simulating the hopf model
def sim_hopf(hopf_frequencies, SC, G, **pars):
    """
    Parameters 
    ----------
    hopf_frequencies : the hopf frequencies per node, in format (n_roi)
    SC : the structural connectivity (or updated EC), in format (n_roi, n_roi)
    G : the global coupling factor, as a float
    
    Example of pars:
        pars = {'n_runs':10, 'n_roi':90, 'TR':0.1, 'dt':0.025, 'n_time': 50}

    Returns
    -------
    The simulated timeseries in format (n_runs, n_roi, n_time/TR)
    """
    

    # as inputs:
        # hopf frequencies
        # G
        # SC
        # default is perturbing all nodes
        # pars is a dictionary containing:
            # n_runs
            # n_roi
            # TR
            # dt
            # n_time

    n_runs = pars['n_runs'] #100
    n_roi = pars['n_roi'] #90 
    TR = pars['TR'] # 0.1
    dt = pars['dt'] # 0.025
    TIME_POINTS = pars['n_time']/TR
    TIME_POINTS = int(TIME_POINTS)
    # pert_ampl = pars['pert_ampl']

    sig         = 6e-4
    dsig        = np.sqrt(dt) * sig      # Scaling factors for the noise!

    tss = np.zeros((n_runs, n_roi, TIME_POINTS))
    
    omega = np.matlib.repmat(2 * np.pi * hopf_frequencies.T, 1, 2)     # Node frequencies
    omega[:, 0] *= -1                                               # The frequency associated with the x equation is negative
    weighted_conn = G * SC                                          # for how the integration is computed
    sum_conn = np.matlib.repmat(weighted_conn.sum(1, keepdims=True), 1, 2)          
    for subsim in range(n_runs):
        # Initialize variables
        z = 0.1 * np.ones((n_roi, 2))         # x = z[:, 0], y = z[:, 1]
        xs = np.zeros((TIME_POINTS, n_roi))   # Array to save data
        nn = 0                                      # Number of simulated values saved
    
        # Discard the first 2k seconds (transient)
        for t in np.arange(0, 2000+dt, dt):
            a           = -0.02 * np.ones((n_roi, 2))  # Bifurcation parameter¡
            zz          = z[:, ::-1]                         # flipped so that zz[:, 0] = y; zz[:, 1] = x
            interaction = weighted_conn @ z - \
                sum_conn * z                        # sum(Cij*xi) - sum(Cij)*xj
            bifur_freq  = a * z + zz * omega         # Bifurcation factor and freq terms
            intra_terms = z * (z*z + zz*zz)
            
            # Gaussian noise
            noise       = dsig * np.random.normal(size=(n_roi, 2))
         
            z           = z + dt * (bifur_freq - intra_terms + interaction) + noise
    
             
        # Compute and save the non-transient data (x = BOLD signal (interpretation), y = some other osc)
        # The way it has been impleneted here is conservative for the number of points saved
        iter0 = 0
        while nn < TIME_POINTS:
            zz = z[:, ::-1]  # flipped so that zz[:, 0] = y; zz[:, 1] = x
            interaction = weighted_conn @ z - \
                sum_conn * z  # sum(Cij*xi) - sum(Cij)*xj
            intra_terms = z * (z*z + zz*zz)
    
            # Gaussian noise
            noise = dsig * np.random.normal(size=(n_roi, 2))
      
            # Integrative step without perturbation
            a           = -0.02 * np.ones((n_roi, 2))  
    
            bifur_freq = a * z + zz * omega  # Bifurcation factor and freq terms
            z = z + dt * (bifur_freq - intra_terms + interaction) + noise
            iter0 += 1
            # Save simulated data if conditions are met
            # if t % TR < (dt*TR)/5:
            if iter0 >= TR/dt:      # Only saving when the simulated value is at the timepoint falling on a TR point
                iter0 = 0
                xs[nn, :] = z[:, 0].T   # save values from x
                nn += 1
    
        # Get the timeseries with parcells as rows
        # Save timeseries (ts = xs.T) with the rest of the subsims
        tss[subsim, :, :] = xs.T
    return tss
    


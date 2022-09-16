# module for simulating distributions (mainly for testing the kendall tau statistic)

import numpy as np
import pandas as pd
from typing import *

###############################
###### POWER LAW SAMPLING #####
###############################

def rndm(a: float, b: float, power: float, size: int = 1, seed: int = 0) -> np.ndarray:
    '''
    Power-law gen for $pdf(x) \propto x^power$ for a <= x <= b
    
    ------------
    INPUT:
    ------------
    a: lower limit of domain
    b: upper limit of domain
    power: power in PDF of power law, pdf(x) \propto x^power
    size: size of random sample (default = 1).
    seed: random seed put into numpy (default = 0).
    
    ------------
    OUTPUT:
    ------------
    1D np array with data point(s) randomly generated from the specified power law distribution
    '''
    if power >= -1:
        raise ValueError('g cannot be >= -1')
    
    power += 1
    rng = np.random.default_rng(seed)
    r = rng.random(size)
    ag, bg = a**power, b**power
    return (ag + (bg - ag)*r)**(1/power)

def get_expected(l: np.ndarray, r: np.ndarray, a: float, b: float, power: float, n: int) -> np.ndarray:
    '''
    gets expected # of points in power law distr with l and r edges to bins
    
    ------------
    INPUT:
    ------------
    l: left edge(s) of bins
    r: right edge(s) of bins
    a: lower limit of domain. should be equal to min(l)
    b: upper limit of domain. should be equal to max(r)
    power: exponent in PDF of power law
    n: total number of data points
    
    ------------
    OUTPUT:
    ------------
    np array with expected number of data points for each bin
    '''
    return (l**(power+1) - r**(power+1)) * n / (a**(power+1) - b**(power+1))

##################################
##### REALISTIC FRB SAMPLING #####
##################################

def frb_sample(zparams: List[float] = [0.9, 0.6], Lparams: List[float] =[1e23, 1e26, -2], size: int = 200, seed: int = 0, pdSeries: Union[None, pd.Series] = None):
    '''
    Generates an FRB sample with lognormal redshift distribution and power law luminosity distribution,
    without a luminosity evolution.
    
    ------------
    INPUT:
    ------------
    zparams: [mean, sigma] for lognormal z distribution (default = [0.9, 0.6]).
    Lparams: [initial min, initial max, power for power law] for L distribution (default = [1e23, 1e26, -2]).
    size: size of sample (default = 200).
    seed: random seed put into numpy (default = 0).
    pdSeries: can use a pandas Series to specify parameters (default = None).
    
    ------------
    OUTPUT:
    ------------
    z: random redshift data, generated according to specified lognormal distribution
    L: random luminosity data, generated according to specified power law
    '''
    
    if pdSeries is not None: #overrides all other settings
        #TODO: implement pdSeries random sample, if necessary
        pass
    
    #Generate Data
    rng = np.random.default_rng(seed)
    z = rng.lognormal(mean=zparams[0], sigma=zparams[1], size=size)+0.1
    L = rndm(Lparams[0], Lparams[1], Lparams[2], size=size, seed=seed)
    
    return z, L
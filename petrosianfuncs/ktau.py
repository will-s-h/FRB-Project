#import numpy as np # --> already imported by cosmology
import pandas as pd  
from cosmology import *
import deprecated_cosmology as d

print("hi")

#####################################################
#####################################################
################### g FUNCTIONS #####################
#####################################################
#####################################################

def g(z: Union[float, np.ndarray], k: float) -> Union[float, np.ndarray]:
    '''
    ----------I/O-----------
    INPUT: 
    z — redshift (single value or np array)
    k — power (float)
    
    OUTPUT: g(z,k), the value of the 'transform' function (no official name for it)
    ------------------------
    
    Notes:
    This is the simpler form of the 'transform' function
    '''
    return (1+z)**k

def g_new(z: Union[float, np.ndarray], k: float) -> Union[float, np.ndarray]:
    '''
    ----------I/O-----------
    INPUT: 
    z — redshift (single value or np array)
    k — power (float)
    
    OUTPUT: g_new(z,k), the value of the 'transform' function (no official name for it)
    ------------------------
    
    Notes:
    This is the improved form of the 'transform' function
    '''
    Z = z+1
    Z_c = 3.5
    return Z**k/(1+Z/Z_c)**k

#####################################################
#####################################################
################### KENDALL TAU #####################
#####################################################
#####################################################

def ktau(Es: np.ndarray, z: np.ndarray, Flim: float, func: Callable = E_v, g:float = g, k: float = 0, params: List[float]=[1.5], diag: bool = False):
    '''
    ----------I/O-----------
    INPUT: 
    Es — energies (np array)
    z — redshifts (np array)
    Flim — limiting fluence (float)
    func — limiting function [default=E_v]
    g — 'transform' function [default=g]
    k — power (float) [default=0]
    params — parameters for limiting function, [default = [1.5]]
    diag — return R, E, and V (boolean) [default=False]
    
    OUTPUT: the kendall tau value at k
    ------------------------
    
    Notes:
    should give same value as ktau_E
    '''
    
    R = []
    E = []
    V = []
    T = []
    
    gs = g(z,k)
    Elims_ = func(Flim, z, *params)
    
    for i in range(len(Es)):
        higher = 0
        lower = 0
        
        if Es[i] <= Elims_[i]: #skip points under cutoff
            continue
        
        for j in range(len(Es)): 
            
            if j == i or Es[j] <= Elims_[j]: #skip over itself or points under cutoff
                continue
                
            if(Es[j]/gs[j] >= Elims_[i]/gs[i] and z[j] <= z[i]): #associated set requirements
                
                if(Es[j]/gs[j] > Es[i]/gs[i]):
                    higher += 1
                else:
                    lower += 1
        
        N_i = higher+lower+1.
        R_i = (lower+1.)/N_i
        E_i = (1.+1./N_i)/2.
        V_i = (1.-1./N_i**2.)/12.
        
        if V_i == 0: #essentially, ignore points with only themselves in the associated set
            V_i += 1
        
        R.append(R_i) #diagnostic prints: print(f'loc: ({Es[i], z[i]}), high/low: {higher, lower}, rank: {R[len(R)-1]}'), print(f'loc: ({Es[i], z[i]}), {R_i}, {E_i}, {V_i}')
        E.append(E_i) #0.5
        V.append(V_i) #1/12
        T.append((R_i-E_i)/(V_i)**0.5)
    
    #numer = sum(R) - sum(E)
    #denom = (sum(V))**0.5
    if diag:
        df = pd.DataFrame()
        df['R'] = R
        df['E'] = E
        df['V'] = V
        return df
    return sum(T)/(len(T))**0.5

def ktau_E(Es, Elims, z, g=g, k=0):
    '''
    ----------I/O-----------
    INPUT: 
    Es — energies (np array)
    Elims — limiting energies (np array)
    z — redshifts (np array)
    g — 'transform' function [default=g]
    k — power (float) [default=0]
    
    OUTPUT: 
    The kendall tau value at k
    ------------------------
    
    Notes:
    should give same value as ktau
    '''
    
    R = []
    E = []
    V = []
    T = []
    
    gs = g(z,k)
    
    for i in range(len(Es)):
        higher = 0
        lower = 0
        
        if Es[i] <= Elims[i]: #not considering data points under the cutoff
            continue
        
        for j in range(len(Es)): 
            if j == i or Es[j] <= Elims[j]: #skip over itself, data points under cutoff
                continue
                
            if(Es[j]/gs[j] >= Elims[i]/gs[i] and Elims[j] <= Elims[i]): #associated set requirements
                
                if(Es[j]/gs[j] > Es[i]/gs[i]):
                    higher += 1
                else:
                    lower += 1
        
        N_i = higher+lower+1.
        R_i = (lower+1.)/N_i
        E_i = (1.+1./N_i)/2.
        V_i = (1.-1./N_i**2)/12.
        
        if V_i == 0: #essentially, ignore points with only themselves in the associated set
            V_i += 1
        
        R.append(R_i) #diagnostic print: print(f'loc: ({Es[i], z[i]}), high/low: {higher, lower}, rank: {R[len(R)-1]}')
        E.append(E_i) #0.5
        V.append(V_i) #1/12
        T.append((R_i-E_i)/V_i**0.5)
    
    #numer = sum(R) - sum(E)
    #denom = (sum(V))**0.5
    return sum(T)/(len(T))**0.5

#####################################################
#####################################################
################## ANALYSIS FUNCS ###################
#####################################################
#####################################################

def sigma_z(Es, z, Flim, func=E_v, g=g, k=0, params=[1.5]): #z0 being the z of interest
    '''
    WARNING: REQUIRES Es AND zs TO BE SORTED by zs!!!!!
    
    ----------I/O-----------
    INPUT: 
    Es — energies (np array)
    z — redshifts (np array)
    Flim — limiting fluence (float)
    func — limiting function [default=E_v]
    g — 'transform' function [default=g]
    k — power (float) [default=0]
    params — parameters for limiting function, [default = [1.5]]
    
    OUTPUT: 
    the dot_sigma(z) values at each eligible value of z
    ------------------------
    
    Notes:
    N/A
    '''
    
    gs = g(z,k)
    Elims_ = func(Flim, z, *params)
    
    zs_actual = []
    ms = []
    
    for i in range(len(Es)):
        m_i = 0

        if(Es[i] <= Elims_[i]): #not considering data points under the cutoff
            continue
        
        for j in range(len(Es)): 
            if i == j: #exclude itself
                continue
            if(Es[j]/gs[j] >= Elims_[i]/gs[i] and z[j] <= z[i]): #associated set requirements
                m_i += 1
                
        zs_actual.append(z[i])
        ms.append(m_i)
    
    #note that in sigma formula, 1+1/m[0] should be replaced with 1
    sigmas = [1]
    
    for i in range(1, len(zs_actual)):
        s_i = sigmas[i-1]*(1.+1./ms[i])
        sigmas.append(s_i)
    
    return np.array(sigmas), np.array(zs_actual)

z_Emodel = np.linspace(0, 3, 10000)
E_Emodel = E_v(2.0, z_Emodel, alpha=1.5)

def phi_E(Es, z, Flim, func=d.z_E_V1, g=g, k=0, params=[1.5], model=[E_Emodel, z_Emodel]):
    #Note: we assume that Es, z are clean already (i.e. no points under the cutoff)
    #no need to be sorted, already sorted within this function
    
    gs = g(z,k)
    Emod, zmod = model
    Es_overg, z = sort_by_first(Es/gs, z)
    zmax = func(Es_overg, Emod/g(zmod, k), zmod)
    
    # DEBUG PLOT:
#     plt.scatter(z, Es_overg, marker='.')
#     plt.yscale('log')
#     P=-50
#     plt.scatter(z[P], Es_overg[P], s=200)
#     plt.hlines(Es_overg[P], 0, zmax[P])
#     print(zmax[P])
#     plt.plot(z_Emodel[100:], E_Emodel[100:])
    
    E_primes = []
    Es_raw = []
    ns = []
    
    for i in reversed(range(len(Es))):
        n_i = 0
        
        for j in range(len(Es)):
            if i == j: #skip over itself
                continue
                
            if (Es_overg[j] >= Es_overg[i] and z[j] <= zmax[i]): #associated set requirements
                n_i += 1
        
        ns.append(n_i)
        E_primes.append(Es_overg[i])
        Es_raw.append(Es_overg[i]*g(z[i],k))
    
    phis = [1]
    
    for i in range(1, len(E_primes)):
        phi_i = phis[i-1]*(1.+1./ns[i])
        phis.append(phi_i)
        
    return np.array(phis), np.array(E_primes), np.array(Es_raw)

print("done")
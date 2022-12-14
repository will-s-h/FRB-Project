# module for functions related to the kendall tau statistic

import numpy as np
import cosmology as c
from typing import *
from queue import Queue
from sortedcontainers import SortedList
from bisect import bisect_right

#################################
########### CONSTANTS ###########
#################################

#default values for dm_ex
_DM_ISM: Final[float] = 30. #dispersion measure for interstellar medium
_DM_HALO: Final[float] = 50. #dispersion measure for the halo

#default value for k in get_z_story
_K_MACQUART: Final[float] = 700/0.7

#default values for evolution functions
_Z_CR: Final[float] = 3.5
_SHARPNESS: Final[float] = 3.

#################################
####### DM-z RELATIONSHIP #######
#################################

#remove excess DM
def dm_ex(dm_tot: Union[float, np.ndarray], dm_ism: Union[float, np.ndarray] = _DM_ISM, dm_halo: Union[float, np.ndarray] = _DM_HALO) -> Union[float, np.ndarray]:
    '''
    Removes DM_ISM and DM_HALO from DM. Assumes values of 30 and 50 pc cm^-3, respectively.
    '''
    return dm_tot - (dm_ism + dm_halo)

#Story's get_z function
def get_z_story(dm: Union[float, np.ndarray], k: float = _K_MACQUART) -> Union[float, np.ndarray]:
    '''
    Story's get_z function. Solves the quadratic equation shown in the above Markdown to get z(DM).
    Almost like a linear approximation for DM.
    '''
    if type(dm) is np.ndarray:
        arr = np.zeros(len(dm))
        for i, dm_i in enumerate(dm):
            arr[i] = get_z_story(dm_i)
        return arr
    
    y = dm_ex(dm)
    arr = np.roots([1, (1 - y/k), ((1/k)*(50 - y))]) # apply Macquart relation
    zeroes = np.argwhere(arr > 0).flatten()
    if len(zeroes) > 0:
        return arr[zeroes].item()
    else:
        return 0

#my get_z function
def get_z(dm: Union[float, np.ndarray], func: Union[str, Callable[[float], float]] = "Arcus", fast: bool = False) -> Union[float, np.ndarray]:
    '''
    My get_z function. Allows for both Arcus and Zhang modes.
    Zhang mode is preferred, since it is continuous and more accurate.
    fast mode allows for faster calculation (but less accuracy), using linear interpolation of precalculated array.
    '''
    y = dm_ex(dm)
    return c.z_DM(y, func=func, fast=fast)

#################################
####### FL <-> LUMINOSITY #######
#################################

def get_L(fluence: Union[float, np.ndarray], rs: Union[float, np.ndarray], alpha: float = 1) -> Union[float, np.ndarray]:
    '''
    Assumes fluence is inputted in units of Jy ms.
    Returns L in the units of J/Hz. Really more like energy than luminosity.
    Alpha = 1 accounts for time dilation. Additional changes to alpha also includes a k-correction.
    '''
    return c.E_v(fluence, rs, alpha=alpha)


#################################
###### LUMINOSITY EVOLUTION #####
#################################

def g_story(z: Union[float, np.ndarray], k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    Story's evolution function
    '''
    Z = 1+z
    return np.piecewise(Z, [Z <= _Z_CR, Z > _Z_CR], [lambda x: x**k, lambda x: x**k * (1 + _Z_CR**k) / (x**k + _Z_CR**k)])

def g_simple(z: Union[float, np.ndarray], k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return (1+z)**k

def g_complex(z: Union[float, np.ndarray], k: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    Z = 1+z
    n = _SHARPNESS
    return Z**k/(1+(Z/_Z_CR)**n)**(k/n)

#################################
####### KENDALL TAU FUNCS #######
#################################

def truncate(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return x[y > x], y[y > x], z[y > x]

def ktau_fast(y: np.ndarray, x: np.ndarray, z: np.ndarray, gfunc: Callable[[np.ndarray, float], np.ndarray] = g_complex, k:float = 0, debug: bool = False) -> float:
    '''
    ktau, but only for default parameters.
    performs analysis in O(N log N).
    '''
    
    x, y, z = truncate(x, y, z) 
    gs = gfunc(z, k)
    xevo, yevo = x/gs, y/gs
    sortedx, sortedy = c.sort_by_first(xevo, yevo)
    
    todelete = [[] for i in range(len(sortedx)+1)]
    toadd = Queue() # queue of {y, x} pairs
    S = SortedList() # contains sorted list of luminosities currently in the sorted set

    ns = np.zeros(len(sortedx))
    rnks = np.zeros(len(sortedx))

    for i in range(len(sortedx)): # O(N log N)
        # permanently add all FRBs with L_min < Lmin,i
        while not toadd.empty() and toadd.queue[0][1] < sortedx[i]:
            pair = toadd.get()
            S.add(pair[0])
        
        # delete all FRBs with L < Lmin,i
        for todel in todelete[i]:
            S.discard(todel)
        
        # temporarily add this current FRB into the set
        S.add(sortedy[i])
        ranklo, rankhi = S.bisect_left(sortedy[i]), S.bisect_right(sortedy[i])-1
        rank = (ranklo+rankhi)/2 + 1
        rnks[i] = rank
        ns[i] = len(S)
        S.discard(sortedy[i])
        
        if debug:
            print(ns[i], rnks[i])
            
        # place the toadd marker for this FRB
        toadd.put((sortedy[i], sortedx[i]))

        # place the todelete marker for this FRB
        todelete[bisect_right(sortedx, sortedy[i])].append(sortedy[i])

    rnks = rnks[ns > 1] #excludes ns = 1 cases
    ns = ns[ns > 1]

    _T = (rnks-.5*(ns+1))/np.sqrt((ns**2 - 1)/12)
    return np.sum(_T)/(len(_T))**0.5

def ktau_E(_y: np.ndarray, _x: np.ndarray, _z: np.ndarray, gfunc: Callable[[np.ndarray, float], np.ndarray] = g_complex, k: float = 0, params: Union[None, str, list[int]] = None, debug: bool = False) -> float:
    '''
    INPUT:
    _y = Es
    _x = Elims
    _z = z (redshift)
    gfunc = g(z, k)
    k = coefficient in gfunc
    params:
        params[0]: dealing with n_i = 1
            0 (default): exclude data points with n_i = 1 (same as story's)
            1 (old): use n_i = 1 data points as T_i = 0 points.
        params[1]: dealing with ties in rank
            0 (default): rank is averaged across ties (paper suggested method)
            1 (old): maximum rank is used
        params[2]: dealing with upper boundary in x
            0 (default): x < x[i] and data point i are in associated set (paper suggested method)
            1 (old): x <= x[i] is in associated set
        params[3]: dealing with lower boundary in y
            0 (old, default): y >= x used (paper suggested)
            1 (story's): y > x used in associated set
    debug: default is False. if True, will print ns and rnks.
    
    some special inputs:
    None -> params = [0,0,0,0] (default)
    "story" -> params = [0,0,0,1] (story's ktau function)
    "old" -> params = [1,1,1,0] (my old ktau function)
    '''
    if params is None or params == "default":
        params = [0,0,0,0]
    elif params == "story":
        params = [0,0,0,1]
    elif params == "old":
        params = [1,1,1,0]
    
    x, y, z = truncate(_x, _y, _z)
    gs = gfunc(z, k)
    xevo, yevo = x/gs, y/gs
    
    ns = np.zeros(len(yevo))
    rnks = np.zeros(len(yevo))
    includei = np.full(len(yevo), False)
    
    for i in range(len(yevo)):
        includei[i] = True
        mask = ((yevo > xevo[i]) if params[3] == 1 else (yevo >= xevo[i])) & ((x <= x[i]) if params[2] == 1 else (x < x[i]) | includei) # all instances of x changed to xevo
        includei[i] = False
        
        ns[i] = np.sum(mask)
        rank_lo = np.sum(yevo[mask] < yevo[i]) + 1
        rank_hi = np.sum(yevo[mask] <= yevo[i])
        rnks[i] = rank_hi if params[1] == 1 else (rank_lo + rank_hi)/2
        
        if debug:
            print(ns[i], rnks[i])
        
        if(params[0] == 1 and ns[i] == 1):
            ns[i] += 2
            rnks[i] += 1 #essentially makes R_i - E_i = 0, but still contribute 1 to len(_T)
    
    if(params[0] == 0):
        rnks = rnks[ns > 1] #excludes ns = 1 cases
        ns = ns[ns > 1]
    
    _T = (rnks-.5*(ns+1))/np.sqrt((ns**2 - 1)/12)
    return np.sum(_T)/(len(_T))**0.5

def binary_search_k(minL, L, z, truek, func=ktau_fast, precision=1e-3):
    lo = truek-10
    hi = truek+10
    
    while hi - lo > precision:
        kmid = (hi+lo)/2
        taumid = func(L, minL, z, gfunc=g_complex, k=kmid)
        
        if taumid > 0:
            lo = kmid
        elif taumid < 0:
            hi = kmid
        else:
            return kmid
    
    return (hi+lo)/2
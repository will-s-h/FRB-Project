# module for functions related to cosmology and cosmological models

import numpy as np
from typing import *

#####################################################
#####################################################
#################### CONSTANTS ######################
#####################################################
#####################################################

C: Final[float] = 299792458.0 #speed of light, in m/s
KMSMPC_TO_S: Final[float] = 3.24077929e-20 #convert km/s/Mpc to s^-1
JYMS: Final[float] = 1.0e-29 #1 Jy ms = 1.0e-29 J/(m^2 * Hz)
M_P: Final[float] = 1.6726219e-27 #mass of a proton, in kg
G: Final[float] = 6.67408e-11 #universal gravitational constant in SI units. This value is kept for legacy reasons, recent measurement indicates 6.67430e-11
PCCM3_TO_M2: Final[float] = 3.08567758e22 #convert pc cm^-3 to m^-2
#GYR: Final[float] = 3.1556926e16 #number of seconds in 10^9 years, deprecated

#####################################################
#####################################################
################ COSMOLOGICAL MODEL #################
#####################################################
#####################################################

COSM_MODEL: List[float] = [0.286, 0.714, 0.049, 70] #Omega_M (matter), Omega_Lambda (dark energy), Omega_b (baryons), H_0 (Hubble constant, in km/s/Mpc)
O_M: float; O_L: float; O_B: float
O_M, O_L, O_B = COSM_MODEL[0], COSM_MODEL[1], COSM_MODEL[2] #Omega_M, Omega_Lambda, and Omega_b
H_0: float = COSM_MODEL[3]*KMSMPC_TO_S #convert km/s/Mpc to s^-1

def set_default() -> None:
    '''
    ------------------------
    INPUT: N/A
    OUTPUT: N/A
    ------------------------
    
    Notes:
    Sets COSM_MODEL, O_M, O_L, O_B, and H_0 to default values.
    '''
    global COSM_MODEL, H_0, O_M, O_L, O_B
    COSM_MODEL = [0.286, 0.714, 0.049, 70]
    O_M, O_L, O_B = COSM_MODEL[0], COSM_MODEL[1], COSM_MODEL[2]
    H_0 = COSM_MODEL[3]*KMSMPC_TO_S

def update_cmodel(new_model: Union[List[float], np.ndarray]) -> None:
    '''
    ------------------------
    INPUT: new_model - new cosmological model parameters (list or np.ndarray)
    OUTPUT: N/A
    ------------------------
    
    Notes:
    This function updates COSM_MODEL, O_M, O_L, O_B, and H_0.
    '''
    if type(new_model) is not np.ndarray and type(new_model) is not list:
        raise TypeError("Type not compatible with COSM_MODEL. Update failed.")
    
    if len(new_model) != 4:
        raise ValueError("Length of list or np array is incompatible with COSM_MODEL. Update failed.")
    
    global COSM_MODEL, H_0, O_M, O_L, O_B
    COSM_MODEL = new_model[:] #makes new copy with [:]
    O_M, O_L, O_B = COSM_MODEL[0], COSM_MODEL[1], COSM_MODEL[2]
    H_0 = COSM_MODEL[3]*KMSMPC_TO_S

#####################################################
#####################################################
################# HELPER FUNCTIONS ##################
#####################################################
#####################################################

#VERSION 2
def sort_by_first(list1: Union[list, np.ndarray], list2: Union[list, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    '''
    ------------------------
    INPUT: list1, list2 (two lists/np arrays)
    OUTPUT: list1, list2 (numpy arrays) sorted in increasing order of list1
    ------------------------
    '''
    zipped_lists = zip(list1, list2)
    sorted_pairs = sorted(zipped_lists, key=lambda x:x[0]) #sort by first element, sort in place if first element equal

    tuples = zip(*sorted_pairs)
    list1, list2 = [list(tuplee) for tuplee in tuples]
    return np.array(list1), np.array(list2)

from scipy import integrate

def integrate_mult_ends(func: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]], begin: float, end: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    ------------------------
    INPUT: 
    func ??? function to integrate
    begin - begin value (single value)
    end - end value (single value or np.ndarray)
    
    OUTPUT: 
    Values of definite integrals from begin to end value(s), using scipy.integrate.quad.
    ------------------------
    
    Notes:
    Runs in roughly O(N log N), where N is the number of end values.
    '''
    if type(end) is not np.ndarray:
        return integrate.quad(func, begin, end)[0]
    
    inds = np.arange(0, len(end))
    sortx, inds = sort_by_first(end, inds)
    integrals = np.zeros(len(end))
    
    integral = 0
    curx = begin
    for i, xi in enumerate(sortx):
        integral += integrate.quad(func, curx, xi)[0]
        integrals[inds[i]] = integral
        curx = xi
    
    return integrals

#####################################################
#####################################################
##################### DISTANCES #####################
#####################################################
#####################################################

def E(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    ------------------------
    INPUT: z ??? redshift (single value or np array)
    OUTPUT: E(z), defined so that H = H_0 * E(z) (provides Hubble parameter at arbitrary z)
    ------------------------
    
    Notes:
    Note that typically in literature, E(z) is defined as (1+z)^3 * O_M + O_L, but this function takes the square root of that expression.
    '''
    
    return ((1+z)**3 * O_M + O_L)**0.5

#VERSION 3
def D_C(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    ------------------------
    INPUT: z ??? redshift (single value or np array)
    OUTPUT: comoving distance, in units of hubble distances (c/H_0)
    ------------------------
    
    Notes:
    Uses default scipy.integrate.quad. Typically very low error values.
    '''
    
    return integrate_mult_ends(lambda x: 1/E(x), 0, z)
    
def D_L(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    ------------------------
    INPUT: z ??? redshift (single value or np array)
    OUTPUT: luminosity distance, in units of hubble distances (c/H_0)
    ------------------------
    
    Notes:
    Uses default scipy.integrate.quad. Typically very low error values.
    '''
    return (1+z)*D_C(z)

#####################################################
#####################################################
############### DISPERSION MEASURES #################
#####################################################
#####################################################

#note: Arcus and Zhang also differ by a multiplicative constant

def dDMdz_Arcus(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]: 
    '''
    ------------------------
    INPUT: z ??? redshift (single value or np array)
    OUTPUT: dDM/dz ??? increase in DM per increase z, in pc cm^-3
    ------------------------
    
    Notes:
    Three separate sections, corresponding to presence/lack of H or He.
    '''
    
    const = 3*C*H_0*O_B/(8*np.pi*G*M_P) # SI units
    fac = 0.75*np.piecewise(z, [z<=8, z>8], [1, 0]) + 0.125*np.piecewise(z, [z<=2.5, z>2.5], [1, 0])
    return (1+z)*(const/PCCM3_TO_M2)*fac/E(z)

def dDMdz_Zhang(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    ------------------------
    INPUT: z ??? redshift (single value or np array)
    OUTPUT: dDM/dz ??? increase in DM per increase z, in pc cm^-3
    ------------------------
    
    Notes:
    One smooth peak with maximum at z<1.
    '''
    
    const = 3*C*H_0*O_B/(8*np.pi*G*M_P) * 7/8 * 0.84
    return (1+z)*(const/PCCM3_TO_M2)/E(z)

#VERSION 2
def DM(z: Union[float, np.ndarray], dDMdz: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]] = dDMdz_Zhang) -> Union[float, np.ndarray]:
    '''
    ------------------------
    INPUT: 
    z     ??? single value or array/list of redshift
    dDMdz ??? dDM/dz model used, default Arcus
    
    OUTPUT: 
    DM(z) ??? DM(z), in pc cm^-3, single value or list depending on z input
    ------------------------
    
    Notes:
    Uses scipy.integrate.quad. Default has now changed to DM_Zhang, due to faster runtime and accuracy of inverse.
    '''
    return integrate_mult_ends(dDMdz, 0, z)

#####################################################
#####################################################
################### INVERSE z(DM) ###################
#####################################################
#####################################################

from pynverse import inversefunc
_z_DM_Arcus: Callable = inversefunc(lambda z: DM(z, dDMdz=dDMdz_Arcus), domain=[0,8])
_z_DM_Zhang: Callable = inversefunc(lambda z: DM(z, dDMdz=dDMdz_Zhang))
import warnings as w

from deprecated_cosmology import z_DM_V1 #circular import, should be okay
_z_low: float = 0.
_z_high: float = 8.
_num_z: int = 2000
_z: np.ndarray = np.linspace(_z_low, _z_high, _num_z)
_DM_Arcus: np.ndarray = DM(_z, dDMdz=dDMdz_Arcus)
_DM_Zhang: np.ndarray = DM(_z, dDMdz=dDMdz_Zhang)

def set_fast_DM(z_low: float = 0., z_high: float = 8., num_z: int = 2000) -> None:
    '''
    Precomputes DM arrays for later use in the fast z_DM version. Only precomputes Arcus and Zhang models.
    ------------------------
    INPUT:
    DM_low - single value indicating lower end of DM range
    DM_high - single value indicating upper end of DM range
    num_DM - single value indicating number of data points calculated between DM_low and DM_high
    
    OUTPUT:
    N/A
    ------------------------
    '''
    
    global _z_low, _z_high, _num_z, _z, _DM_Arcus, _DM_Zhang
    _z_low, _z_high, _num_z = z_low, z_high, num_z
    _z = np.linspace(_z_low, _z_high, _num_z)
    _DM_Arcus = DM(_z, dDMdz=dDMdz_Arcus)
    _DM_Zhang = DM(_z, dDMdz=dDMdz_Zhang)


def z_DM(DM: Union[float, np.ndarray], func: Union[str, Callable[[float], float]] = "Zhang", fast: bool = False) -> Union[float, np.ndarray]:
    '''
    ------------------------
    INPUT:
    DM - single value or array of DMs (pc cm^-3)
    func - string or callable function 
    
    OUTPUT:
    zs - single value or array of zs, corresponding to DMs
    ------------------------
    
    Notes:
    Changed default to Zhang due to accuracy of inverse.
    '''
    
    if fast:
        #TODO: #2 add fast implementation using precomputed array and simple linear interpolation
        if func != "Arcus" and func != "Zhang":
            print("unknown function. defaulting to slow z_DM().")
        else:
            _DM = _DM_Arcus if func == "Arcus" else _DM_Zhang
            return z_DM_V1(DM, _DM, _z)
    
    invfunc: Callable
    
    if func == "Arcus":
        invfunc = _z_DM_Arcus
    elif func == "Zhang":
        invfunc = _z_DM_Zhang
    else:
        invfunc = inversefunc(func)
    
    with w.catch_warnings():
        w.simplefilter("ignore") #ignore np.bool warning
        if type(DM) != np.ndarray:
            return float(invfunc(max(DM, 0)))
        return invfunc(np.maximum(DM, np.zeros(len(DM))))

#####################################################
#####################################################
####################### ENERGY ######################
#####################################################
#####################################################

def E_v(Fs: Union[float, np.ndarray], z: Union[float, np.ndarray], alpha: float) -> Union[float, np.ndarray]:
    '''
    ------------------------
    INPUT: 
    Fs - fluence, in Jy ms
    z - redshift (single value or list [in increasing order])
    alpha - spectral index (F_v is proportional to v^-a)
    
    OUTPUT: 
    Energy at a frequency E_v (in J/Hz) (single val/list) corresponding with fluence, at z
    ------------------------
    '''
    return Fs*4*np.pi*(D_L(z)*C/H_0)**2*JYMS/((1+z)**(2-alpha))

_used: Dict[float, Callable[[Union[float, np.ndarray]], np.ndarray]] = {}

def reset_used() -> None:
    '''
    Clears dictionary of inverse functions for E_v(z).
    '''
    global _used
    _used = {}

#### NOT TESTED
def z_E(E_i: Union[float, np.ndarray], F: float, alpha: float) -> Union[float, np.ndarray]:
    '''
    WARNING: may have issues if E(z) is non-monatonic 
    
    ------------------------
    INPUT: 
    E_i - a single value or np.array of Es
    Es - a pre-calculated & sorted list of Es
    zs - a pre-calculated & sorted list of zs, that align one-to-one with Es
    
    OUTPUT: 
    zs - linearly interpolated list of zs that correspond with the input E
    ------------------------
    '''
    
    f: Callable[[Union[float, np.ndarray]], float]
    
    if round(alpha, 7) in _used:
        f = _used[round(alpha, 7)]
    else:
        f = inversefunc(lambda x: E_v(F, x, round(alpha, 7))/F)
        _used[round(alpha, 7)] = f
    
    if type(E_i) != np.ndarray:
        return float(f(E_i))
    
    return f(E_i)
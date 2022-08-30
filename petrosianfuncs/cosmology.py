import numpy as np
from typing import *

#####################################################
#####################################################
#################### CONSTANTS ######################
#####################################################
#####################################################

c: Final[float] = 299792458.0 #speed of light, in m/s
kmsMpc_to_s: Final[float] = 3.24077929e-20 #convert km/s/Mpc to s^-1
JYMS: Final[float] = 1.0e-29 #1 Jy ms = 1.0e-29 J/(m^2 * Hz)
M_P: Final[float] = 1.6726219e-27 #mass of a proton, in kg
G: Final[float] = 6.67408e-11 #universal gravitational constant in SI units. This value is kept for legacy reasons, recent measurement indicates 6.67430e-11
#GYR: Final[float] = 3.1556926e16 #number of seconds in 10^9 years, deprecated

#####################################################
#####################################################
################ COSMOLOGICAL MODEL #################
#####################################################
#####################################################

cosm_model = [0.286, 0.714, 0.049, 70] #Omega_M (matter), Omega_Lambda (dark energy), Omega_b (baryons), H_0 (Hubble constant, in km/s/Mpc)
O_M, O_L, O_B = cosm_model[0], cosm_model[1], cosm_model[2] #Omega_M, Omega_Lambda, and Omega_b
H_0 = cosm_model[3]*kmsMpc_to_s #convert km/s/Mpc to s^-1

def set_default() -> None:
    '''
    ------------------------
    INPUT: N/A
    OUTPUT: N/A
    ------------------------
    
    Notes:
    Sets cosm_model, O_M, O_L, O_B, and H_0 to default values.
    '''
    global cosm_model, H_0, O_M, O_L, O_B
    cosm_model = [0.286, 0.714, 0.049, 70]
    O_M, O_L, O_B = cosm_model[0], cosm_model[1], cosm_model[2]
    H_0 = cosm_model[3]*kmsMpc_to_s

def update_cmodel(new_model: Union[List[float], np.ndarray]) -> None:
    '''
    ------------------------
    INPUT: 
    new_model - new cosmological model parameters (list or np.ndarray)
    
    OUTPUT: 
    N/A
    ------------------------
    
    Notes:
    This function updates cosm_model, O_M, O_L, O_B, and H_0.
    '''
    if type(new_model) is not np.ndarray and type(new_model) is not list:
        raise TypeError("Type not compatible with cosm_model. Update failed.")
    
    if len(new_model) != 4:
        raise ValueError("Length of list or np array is incompatible with cosm_model. Update failed.")
    
    global cosm_model, H_0, O_M, O_L, O_B
    cosm_model = new_model[:]
    O_M, O_L, O_B = cosm_model[0], cosm_model[1], cosm_model[2]
    H_0 = cosm_model[3]*kmsMpc_to_s

#####################################################
#####################################################
##################### DISTANCES #####################
#####################################################
#####################################################

def E(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    ------------------------
    INPUT: z — redshift (single value or np array)
    OUTPUT: E(z), defined so that H = H_0 * E(z) (provides Hubble parameter at arbitrary z)
    ------------------------
    
    Notes:
    Note that typically in literature, E(z) is defined as (1+z)^3 * O_M + O_L, but this function takes the square root of that expression.
    '''
    
    return ((1+z)**3 * O_M + O_L)**0.5

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
    list1, list2 = [ list(tuplee) for tuplee in  tuples]
    return np.array(list1), np.array(list2)

from scipy import integrate

def D_C(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    ------------------------
    INPUT: z — redshift (single value or np array)
    OUTPUT: comoving distance, in units of hubble distances (c/H_0)
    ------------------------
    
    Notes:
    Uses default scipy.integrate.quad. Typically very low error values.
    '''
    if not isinstance(z, np.ndarray):
        return integrate.quad(lambda x: 1/E(x), 0, z)[0]
    
    ins = np.arange(0, len(z))
    sortz, ins = sort_by_first(z, ins)
    Ds = np.zeros(len(sortz))
    
    integral = 0
    curz = 0
    for i, zi in enumerate(sortz):
        ans = integrate.quad(lambda x: 1/E(x), curz, zi)[0]
        integral += ans
        curz = zi
        Ds[ins[i]] = integral
    
    return Ds
    
def D_L(z: Union[float, np.ndarray], temp: bool = False) -> Union[float, np.ndarray]:
    '''
    (fixed): input order is NOT same as output order
    
    ------------------------
    INPUT: 
    z — redshift (single value or np array)
    temp - boolean indicating to use approximate analytic expression (temporary measure)  
    
    OUTPUT: luminosity distance, in units of hubble distances (c/H_0)
    ------------------------
    
    Notes:
    Uses default scipy.integrate.quad. Typically very low error values.
    temp expression only a good approximation for z<3, but good for testing.
    '''
    
    if temp: #temporary measure
        return 1.56391885121*(z**1.2290110356)
    return (1+z)*D_C(z)


#####################################################
#####################################################
############### DISPERSION MEASURES #################
#####################################################
#####################################################

#note: Arcus and Zhang also differ by a multiplicative constant

def dDMdz_Arcus(z): 
    '''
    ------------------------
    INPUT: z — redshift (single value or np array)
    OUTPUT: dDM/dz — increase in DM per increase z, in pc cm^-3
    ------------------------
    
    Notes:
    Three separate sections, corresponding to presence/lack of H or He.
    '''
    
    C = 3*c*H_0*O_B/(8*np.pi*G*M_P) # SI units
    convert = 3.08567758e22 #1 pc cm^-3 in SI units
    fac = 0.75*np.piecewise(z, [z<=8, z>8], [1, 0]) + 0.125*np.piecewise(z, [z<=2.5, z>2.5], [1, 0])
    return (C/convert)*(1+z)*fac/E(z)

def dDMdz_Zhang(z):
    '''
    ------------------------
    INPUT: z — redshift (single value or np array)
    OUTPUT: dDM/dz — increase in DM per increase z, in pc cm^-3
    ------------------------
    
    Notes:
    One smooth peak with maximum at z<1.
    '''
    
    C = 3*c*H_0*O_B/(8*np.pi*G*M_P)
    C *= (7./8.)*0.84
    convert = 3.08567758e22
    return (1+z)*(C/convert)/E(z)

def DM_single_z(z, dDMdz=dDMdz_Arcus):
    '''
    ------------------------
    INPUT: 
    z     — redshift (single value)
    dDMdz — dDM/dz model used, default Arcus
    
    OUTPUT: 
    DM(z), single value of DM at redshift=z
    ------------------------
    
    Notes:
    z must be >= 0
    dz = 0.001 by default, no parameter to change
    '''
    
    if(z < 0):
        print('z out of range')
        return 0
    
    dz = 0.001
    DM = 0.0
    for z_ in np.arange(0, z, dz):
        DM += (dDMdz(z_) + dDMdz(z_+dz))*dz/2.0
    return DM
    

def DM(z, dDMdz=dDMdz_Arcus):
    '''
    (deprecated warning): input order is NOT same as output order
    ------------------------
    INPUT: 
    z     — single value or array/list of redshift
    dDMdz — dDM/dz model used, default Arcus
    
    OUTPUT: 
    DM(z) — DM(z), in pc cm^-3, single value or list depending on z input
    ------------------------
    
    Notes:
    all values of z must be >= 0
    dz = 0.001 by default, no parameter to change
    '''
    
    if not isinstance(z, np.ndarray):
        return DM_single_z(z, dDMdz)
    
    if(min(z) < 0):
        print('z out of range')
        return z*0
    
    indices = np.arange(0, len(z))
    z_s, indices = sort_by_first(z, indices)
    ind = 0
    DMs = np.array([])
    
    dz = 0.001
    DM = 0.0
    for z_ in np.arange(0, max(z)+2*dz, dz):
        DM += (dDMdz(z_) + dDMdz(z_+dz))*dz/2.0
        while(ind < len(z_s) and abs(z_s[ind]-z_) < dz):
            DMs = np.append(DMs, DM)
            ind += 1
    indices, DMs = sort_by_first(indices, DMs)
    return DMs

#####################################################
#####################################################
################### INVERSE z(DM) ###################
#####################################################
#####################################################

from bisect import bisect_left as bleft
#behavior:
# if exactly at an element, returns that index
# if between two elements, returns index of right hand element
# if beyond range, returns right hand index (0 or len(arr))

from bisect import bisect_right as bright
#behavior:
#if exactly at an element, returns index+1
#if between two elements, returns right hand index
# if beyond range, returns right hand index (0 or len(arr))


def z_DM_single(DM, DMs, zs, E_setting=False): #assumes DMs and zs already sorted
    '''
    ------------------------
    INPUT: 
    DM - a single value
    DMs - a pre-calculated & sorted list of DMs
    zs - a pre-calculated & sorted list of zs, that align one-to-one with DMs
    E_setting - True means returning inf if DM > max(DMs); default is False
    
    OUTPUT: 
    zs - linearly interpolated z that correspond with the input DM
    ------------------------
    '''
    if E_setting and DM > max(DMs):
        return np.inf
    
    if(DM > max(DMs) or DM < min(DMs)):
        print(('E' if E_setting else 'DM') + 'out of range: ' + str(DM) + (' J/Hz' if E_setting else ' pc cm^-3'))
        return 0
    
    i = bleft(DMs, DM)
    
    if (i != bright(DMs, DM)): #means that DM is exactly within DMs
        return zs[i]
    
    #if not exactly equal to one of data points, linearly interpolate
    return zs[i-1] + (zs[i]-zs[i-1])*(DM-DMs[i-1])/(DMs[i]-DMs[i-1])

def z_DM(DM_i, DMs, zs, E_setting=False):
    '''
    ------------------------
    INPUT: 
    DM_i - a single value or np.array of DMs
    DMs - a pre-calculated & sorted list of DMs
    zs - a pre-calculated & sorted list of zs, that align one-to-one with DMs
    E_setting - True means returning inf if DM > max(DMs); default is False
    
    OUTPUT: 
    zs - linearly interpolated list of zs that correspond with the input DM
    ------------------------
    '''
    
    if type(DM_i) != np.ndarray:
        return z_DM_single(DM_i, DMs, zs, E_setting=E_setting)
    
    z_f = np.zeros(len(DM_i))
    
    for i, DM in enumerate(DM_i):
        z_f[i] = z_DM_single(DM, DMs, zs, E_setting=E_setting)
    
    return z_f

#####################################################
#####################################################
####################### ENERGY ######################
#####################################################
#####################################################


def E_v(Fs, z, alpha):
    '''
    (deprecated warning): input order is NOT same as output order
    
    ------------------------
    INPUT: 
    Fs - fluence, in Jy ms
    z - redshift (single value or list [in increasing order])
    alpha - spectral index (F_v is proportional to v^-a)
    
    OUTPUT: 
    F_v (single val/list) corresponding with E, at z
    ------------------------
    '''
    return Fs*4*np.pi*(D_L(z)*c/H_0)**2*JYMS/((1+z)**(2-alpha))

def z_E(E_i, Es, zs):
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
    return z_DM(E_i, Es, zs, E_setting=True)
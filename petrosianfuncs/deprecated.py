from cosmology import *

##################################
############ DISTANCES ###########
##################################

#VERSION 1 OF D_C
def D_C_old(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    THIS FUNCTION IS DEPRECATED. See D_C(z) for the new standard.
    (fixed) [input order is NOT same as output order]
    
    ------------------------
    INPUT: z — redshift (single value or np array)
    OUTPUT: comoving distance, in units of hubble distances (c/H_0)
    ------------------------
    
    Notes:
    Currently using low precision, dz=0.001
    '''
    #TEST TEST TEST
    dz = 0.001 
    
    if not isinstance(z, np.ndarray):
        #single value
        zs = np.arange(0., z-dz, dz)
        integral = 0
        for z_i in zs:
            integrand = (1/E(z_i) + 1/E(z_i+dz))/2.0
            integral += integrand*dz
        
        #final add (since we've added extra, we subtract dz, then add z-)
        integrand = (1/E(zs[-1]+dz) + 1/E(zs[-1]+2*dz))/2.0
        integral += integrand * (z-(zs[-1]+dz))
        return integral
    
    indices = np.arange(0, len(z))
    sorted_zs, indices = sort_by_first(z, indices)
    
    D_Cs = np.zeros(len(sorted_zs))
    ind = 0
    zs = np.arange(0., max(z)+dz, dz)
    integral = 0
    for z_i in zs:
        integrand = (1/E(z_i) + 1/E(z_i+dz))/2.0
        while(ind < len(sorted_zs) and sorted_zs[ind] < (z_i+dz)):
            D_Cs[ind] = integral + integrand * (sorted_zs[ind] - z_i)
            ind += 1
        integral += integrand*dz
    indices, D_Cs = sort_by_first(indices, D_Cs)
    return D_Cs

# VERSION 2 OF D_C
def D_C_V2(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    THIS FUNCTION IS DEPRECATED. See D_C(z) for the new standard.
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

#TEMPORARY VERSION OF D_L
def D_L_temp(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    Used to be that D_L had an option to pass a bool into the function:
    temp - boolean indicating to use approximate analytic expression (temporary measure)  
    
    temp expression only a good approximation for z<3, but good for testing.
    '''
    return 1.56391885121*(z**1.2290110356)

##################################
###### DISPERSION MEASURES #######
##################################

#DM_SINGLE_Z (NOW FOLDED INTO DM)
def DM_single_z(z: float, dDMdz: Callable[[float], float] = dDMdz_Arcus) -> float:
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

# VERSION 1 OF DM(z, dDMdz)
def DM_V1(z: Union[float, np.ndarray], dDMdz: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]] = dDMdz_Arcus) -> Union[float, np.ndarray]:
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

##################################
####### INVERSE FUNCTIONS ########
##################################

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

def z_DM_single(DM: float, DMs: Union[list, np.ndarray], zs: Union[list, np.ndarray], E_setting:bool = False) -> float: #assumes DMs and zs already sorted
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

def z_DM(DM_i: Union[float, np.ndarray], DMs: Union[list, np.ndarray], zs: Union[list, np.ndarray], E_setting: bool = False) -> Union[float, np.ndarray]:
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

##################################
############# ENERGY #############
##################################

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
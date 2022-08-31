from cosmology import *

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

def D_L_temp(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    Used to be that D_L had an option to pass a bool into the function:
    temp - boolean indicating to use approximate analytic expression (temporary measure)  
    
    temp expression only a good approximation for z<3, but good for testing.
    '''
    return 1.56391885121*(z**1.2290110356)

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
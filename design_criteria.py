import sys
import numpy as np
import os

def latinize(d):
    nsim = d.shape[0]
    order = np.argsort(d,axis=0)
    ranks = np.argsort(order,axis=0)
    return (ranks + 0.5) / nsim

def get_detas(d, periodic):
    '''
    computes a 3-dim matrix of ccordinnate deltas for all pairs of points in 'd'
    ns x ns x nv matrix
    '''
    deltas = np.abs(d[:, np.newaxis, :] - d[np.newaxis, :, :]) # ns x ns x nv matrix of abs coordinate differences
    if periodic:
        deltas = np.minimum(deltas, 1 - deltas)
    return deltas


    
# *** PERIODIC WRAP-AROUND DISCREPANCY ***
def wd2(d):
    ns, nv = d.shape
    deltas = get_detas(d, periodic=False) #ns x ns x nv matrix of abs coordinate differences
    b = np.prod(3/2 - deltas * (1-deltas), axis=2)
    return np.sum(b) / ns ** 2 - (4/3)**nv

# *** CENTERED L2 DISCREPANCY ***
def cl2(d):    
    ns, nv = d.shape
    # design pre-processing (center shift to zero, shrink to one half)
    d = 0.5*(d - 0.5)
    # second term (1d sum of pruducts)
    t2 = np.sum(  np.prod( 1 + np.abs(d) - np.abs( 2*d*d ), axis=1 ) ) *2/ns

    # third term (2d sum of pruducts)
    abs_d2 = np.abs( np.expand_dims(d, axis=1)  -         np.expand_dims(d, axis=0) )
    abs_d1 = np.abs( np.expand_dims(d, axis=1)) + np.abs( np.expand_dims(d, axis=0) )
    b = np.prod(1.0 + abs_d1 - abs_d2 , axis=2) # square matrix of pruducts
    t3 =  np.sum(b) / (ns ** 2)

    return (13./12.)**nv  - t2 + t3

# *** (PERIODIC) MAXIMIN CRITERION ***
def Mm(d, periodic=True):
    nv = d.shape[1]
    deltas = get_detas(d, periodic) #ns x ns x nv matrix of abs coordinate differences
    L = (np.sum(deltas ** 2, axis=2) ** (1/2)) # squared periodic distance for each point pair (ns x ns matrix)
    np.fill_diagonal(L, nv)   # replace zero distances on the main diagonal with nv's (not to be selected as minimum)
    return np.min(L)   # return minimum distance

# *** (PERIODIC) PHI CRITERION ***
def phip(d, periodic=True):
    ns, nv = d.shape
    p = nv +1 #exponent of distances
    deltas = get_detas(d, periodic) #ns x ns x nv matrix of abs coordinate differences
    L = np.sum(deltas ** 2, axis=2) ** (p/2) # squared periodic distance for each point pair
    np.fill_diagonal(L, 1) # replace zero distances on the main diagonal with ones
    return (np.sum(1 / L) - ns) / (ns ** 2 - ns)  # average of off-diagonal members (subtract the diagonal unit entries)

def maxPro(d: np.ndarray, periodic = False) -> float:  # single loop
    """
    Compute the (u)MaxPro criterion for a given design.

    This function calculates the MaxPro or uMaxPro criterion for a given 2D design matrix `x`.
    The MaxPro criterion is used in experimental design to ensure good space-filling properties.
    If `periodic` is set to True, the function computes the uMaxPro criterion, which accounts
    for periodic boundary conditions.

    Args:
        x (np.ndarray): A 2D array of shape (ns, nv) representing the design points.
        ns (int): The number of samples (design points).
        nv (int): The number of variables (dimensions).
        periodic (bool, optional): If True, computes the uMaxPro criterion
            (periodic case). If False, computes the MaxPro criterion
            (non-periodic case). Default is False.

    Returns:
        float: The computed (u)MaxPro criterion value.

    Notes:
        - The MaxPro criterion favors designs that are space-filling by maximizing the
          minimum product of squared distances between points.
        - The uMaxPro variant uses periodic distance calculations.
    """

    ns, nv = d.shape

    maxpro = 0  # Initialize the criterion accumulator

    # Iterate over each design point
    for i in range(ns):
        # Compute the absolute differences between point i and all previous points
        deltas = np.abs(d[i, :] - d[0:i, :])

        if periodic:
            deltas = np.minimum(deltas, 1 - deltas)

        # Square the differences to get squared distances
        dsq = deltas ** 2

        # Compute the reciprocal of the product of squared distances for each pair
        # Sum them up and add to the maxpro accumulator
        maxpro += np.sum(1. / np.prod(dsq, axis=1))

    return maxpro / ns**2
    
# *** common entry ***
def evaluate(d, crit, params=[]):
    nv = d.shape[1]
    ns = d.shape[0]
    if crit.lower() == "cl2":
        return cl2(d,ns,nv)
    elif crit.lower() == "wd2":
        return wd2(d)
    elif crit.lower() == "pmm":
        return pMm(d)
    elif crit.lower() == "pphi":
        return pphi(d)
    else:
        print ("uknown criterion",crit,", terminating")
        exit()

if __name__ == "__main__":
    # tady zkousim volání funkcí pro nějaké designy:

    nv = 2
    ns = 12
    d=np.array([[0.208333333,0.041666667],[0.791666667,0.125],    [0.375,0.208333333],[0.958333333,0.291666667],[0.541666667,0.375],[0.125,0.458333333],[0.708333333,0.541666667],[0.291666667,0.625],[0.875,0.708333333],[0.458333333,0.791666667],[0.041666667,0.875],[0.625,0.958333333]])
    print (wd2(d))
    print (cl2(d))
    print (pphi(d))
    print (Mm(d))

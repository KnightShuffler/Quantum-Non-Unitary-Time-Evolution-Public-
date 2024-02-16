import numpy as np
from numba import njit
#-----------------#
# General Helpers #
#-----------------#

# Error Tolerance
TOLERANCE = 1e-5

@njit
def int_to_base(x:int, b:int, num_digits:int):
    '''
    convert a non-negative integer x to base b and return the digits in a list
    '''
    if x < 0:
        raise ValueError('x must be non-negative')
    digits = np.zeros(num_digits, dtype=np.int32)
    for i in range(num_digits):
        digits[i] = x % b
        x = x//b
    return digits

@njit
def base_to_int(digits, b):
    '''
    convert a list of base b digits to an integer
    '''
    x = 0
    for (i,digit) in enumerate(digits):
        x += digit * (b**i)
    return x

@njit
def exp_mat_psi(mat:np.array, psi:np.array, truncate:int=-1) -> np.array:
    '''
    Calculates exp(mat)|psi> using the Taylor series of exp(mat)
    if truncate == -1, it will calculate the series up until the norm 
    the previous term above the accepted tolerance,
    else, if truncate == k > 0, it will calculate up to the k-th term of
    the Taylor series (mat)^k / k!
    '''
    chi = psi.copy()
    phi = psi.copy()
    i = 1
    while (truncate < 0 and np.linalg.norm(chi) > TOLERANCE) or (truncate >= 0 and i <= truncate) :
        chi = 1/i * (np.dot(mat, chi))
        phi += chi
        i += 1
    return phi

def fidelity(psi, phi):
    '''
    Returns the fidelity between two pure states psi and phi
    F = | <psi|phi> |^2
    '''
    return np.abs(np.vdot(psi, phi))**2

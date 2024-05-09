import numpy as np

from functools import lru_cache
from copy import deepcopy

from .. import Hamiltonian, hm_list_sum, hm_list_tensor, get_identity_hm_list
from ..construction import get_lowerLeft_hm_list, get_lowerRight_hm_list, get_upperLeft_hm_list, get_upperRight_hm_list

ddxKernel_hm_list = [[
    np.array([2],dtype=np.uint32),
    np.array([1.0j],dtype=np.complex128),
    np.array([0],dtype=np.int32)
]]

@lru_cache
def get_ddx_hm_list(num_qbits:int, periodic_bc_flag:bool=False):
    if num_qbits == 1:
        return ddxKernel_hm_list
    lower_left_term = deepcopy(hm_list_tensor(get_lowerLeft_hm_list(1), get_upperRight_hm_list(num_qbits-1), 1, num_qbits-1))
    lower_left_term[0][1] *= -1
    
    periodic_term = None
    if periodic_bc_flag:
        periodic_ur_term = deepcopy(get_upperRight_hm_list(num_qbits))
        periodic_ur_term[0][1] *= -1
        periodic_term = hm_list_sum(
            periodic_ur_term,
            get_lowerLeft_hm_list(num_qbits)
        )

    return hm_list_sum(
        hm_list_tensor(get_identity_hm_list(1), get_ddx_hm_list(num_qbits-1), 1, num_qbits-1),
        lower_left_term,
        hm_list_tensor(get_upperRight_hm_list(1), get_lowerLeft_hm_list(num_qbits-1), 1, num_qbits-1),
        periodic_term
    )

def generateFirstDerivativeHamiltonian1D(num_qbits:int,
                                         dx:float=1.0,
                                         periodic_bc_flag:bool=False
                                         ) -> Hamiltonian:
    hm_list = deepcopy(get_ddx_hm_list(num_qbits, periodic_bc_flag))
    for hm in hm_list:
        for i,amp in enumerate(hm[1]):
            hm[1][i] /= 2.0*dx
    return Hamiltonian(hm_list, num_qbits)

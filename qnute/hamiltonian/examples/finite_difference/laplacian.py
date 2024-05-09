import numpy as np
from functools import lru_cache
from copy import deepcopy

from ...import Hamiltonian, hm_list_sum, hm_list_tensor, get_identity_hm_list
from ...construction import get_lowerLeft_hm_list, get_lowerRight_hm_list, get_upperLeft_hm_list, get_upperRight_hm_list

laplaceKernel_hm_list = [[
    np.array([1,0], dtype=np.uint32),
    np.array([1.0, -2.0], dtype=np.complex128),
    np.array([0],dtype=np.int32)
]]

@lru_cache
def get_laplace1D_hm_list(num_qbits:int):
    if num_qbits == 1:
        return laplaceKernel_hm_list
    return hm_list_sum(
        hm_list_tensor(get_identity_hm_list(1),   get_laplace1D_hm_list(num_qbits-1),  1, num_qbits-1),
        hm_list_tensor(get_lowerLeft_hm_list(1),  get_upperRight_hm_list(num_qbits-1), 1, num_qbits-1),
        hm_list_tensor(get_upperRight_hm_list(1), get_lowerLeft_hm_list(num_qbits-1),  1, num_qbits-1)
    )

@lru_cache
def get_laplace1D_periodic_hm_list(num_qbits:int):
    return hm_list_sum(
        get_laplace1D_hm_list(num_qbits),
        get_lowerLeft_hm_list(num_qbits),
        get_upperRight_hm_list(num_qbits)
    )

@lru_cache
def get_graycode_laplace1D_continuity_term(num_qbits:int):
    assert num_qbits >= 2
    if num_qbits == 2:
        return hm_list_tensor(get_identity_hm_list(1),get_lowerRight_hm_list(1), 1, 1)
    return hm_list_tensor(
        get_identity_hm_list(1),
        hm_list_tensor(get_lowerRight_hm_list(1), get_upperLeft_hm_list(num_qbits-2), 1, num_qbits-2),
        1, num_qbits-1
    )

@lru_cache
def get_graycode_laplace1D_periodic_term(num_qbits:int):
    # assert num_qbits >= 2
    # if num_qbits == 2:
    #     return hm_list_tensor(get_identity_hm_list(1),
    #                           get_upperLeft_hm_list(1), 1, 1)
    return hm_list_tensor(get_identity_hm_list(1), get_upperLeft_hm_list(num_qbits-1), 1, num_qbits-1)

@lru_cache
def get_graycode_laplace1D_hm_list(num_qbits:int):
    if num_qbits == 1:
        return get_identity_hm_list(1)
    return hm_list_sum(
        hm_list_tensor(get_identity_hm_list(1), get_graycode_laplace1D_hm_list(num_qbits-1), 1, num_qbits-1),
        get_graycode_laplace1D_continuity_term(num_qbits)
    )

@lru_cache
def get_graycode_laplace1D_periodic_hm_list(num_qbits:int):
    return hm_list_sum(
        get_graycode_laplace1D_hm_list(num_qbits),
        get_graycode_laplace1D_periodic_term(num_qbits)
    )

def generateLaplaceHamiltonian1D(num_qbits:int, 
                                 dx:float=1.0,
                                 periodic_bc_flag:bool=False,
                                 homogeneous_flag:bool=False):
    hm_list = []
    if not homogeneous_flag:
        if not periodic_bc_flag:
            hm_list = deepcopy(get_laplace1D_hm_list(num_qbits))
        else:
            hm_list = deepcopy(get_laplace1D_periodic_hm_list(num_qbits))
    else:
        if not periodic_bc_flag:
            hm_list = hm_list_tensor(upperLeftKernel_hm_list, get_laplace1D_hm_list(num_qbits-1), 1, num_qbits-1)
            # hm_list = hm_list_tensor(get_laplace1D_hm_list(num_qbits-1),
            #                          [[upperLeftKernel_hm_list[0], upperLeftKernel_hm_list[1], [0]]])
        else:
            hm_list = hm_list_tensor(upperLeftKernel_hm_list, get_laplace1D_periodic_hm_list(num_qbits-1), 1, num_qbits-1)
            # hm_list = hm_list_tensor(get_laplace1D_periodic_hm_list(num_qbits-1),
            #                          [[upperLeftKernel_hm_list[0], upperLeftKernel_hm_list[1], [0]]])

    for hm in hm_list:
        for i,amp in enumerate(hm[1]):
            hm[1][i] /= dx*dx
    return Hamiltonian(hm_list, num_qbits)

def generateLaplacianHamiltonianMultiDim(num_qbits:np.ndarray[int],
                                         dx:np.ndarray[float],
                                         periodic_bc_flags:np.ndarray[bool]
                                         ) -> Hamiltonian:
    ndims = num_qbits.shape[0]
    min_dx = np.min(dx)
    Lap:Hamiltonian = None
    for dim in range(ndims):
        H = generateLaplaceHamiltonian1D(num_qbits[dim], dx[dim]/min_dx, periodic_bc_flags[dim])
        if dim == 0:
            Lap = Hamiltonian.tensor_product_multi(*[Hamiltonian.Identity(num_qbits[j]) if j != dim else H for j in range(ndims)])
        else:
            Lap += Hamiltonian.tensor_product_multi(*[Hamiltonian.Identity(num_qbits[j]) if j != dim else H for j in range(ndims)])
        
    return Lap

def generateGrayCodeLaplacian1D(num_qbits:int,
                                dx:float=1.0,
                                periodic_bc_flag:bool=False,
                                homogeneous_flag:bool=False):
    hm_list = []
    if not homogeneous_flag:
        if not periodic_bc_flag:
            hm_list = get_graycode_laplace1D_hm_list(num_qbits)
        else:
            hm_list = get_graycode_laplace1D_periodic_hm_list(num_qbits)
    else:
        if not periodic_bc_flag:
            hm_list = hm_list_tensor(upperLeftKernel_hm_list,
                                 get_graycode_laplace1D_hm_list(num_qbits-1), 1, num_qbits-1)
        else:
            hm_list = hm_list_tensor(upperLeftKernel_hm_list,
                                 get_graycode_laplace1D_periodic_hm_list(num_qbits-1), 1, num_qbits-1)
    
    for hm in hm_list:
        for i,amp in enumerate(hm[1]):
            hm[1][i] /= dx*dx
    return Hamiltonian(hm_list, num_qbits)

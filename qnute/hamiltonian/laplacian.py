import numpy as np
from functools import lru_cache

from .import Hamiltonian, hm_list_sum, hm_list_tensor

laplaceKernel_hm_list = [
    np.array([1,0], dtype=np.uint32),
    np.array([1.0, -2.0], dtype=np.complex128),
]

lowerLeftKernel_hm_list = [
    np.array([1,2],dtype=np.uint32),
    np.array([0.5, -0.5j], dtype=np.complex128)
]

upperRightKernel_hm_list = [
    np.array([1,2],dtype=np.uint32),
    np.array([0.5, 0.5j], dtype=np.complex128)
]

upperLeftKernel_hm_list = [
    np.array([3,0],dtype=np.uint32),
    np.array([0.5,0.5],dtype=np.complex128)
]

lowerRightKernel_hm_list = [
    np.array([3,0],dtype=np.uint32),
    np.array([-0.5,0.5],dtype=np.complex128)
]

@lru_cache
def get_lowerLeft_hm_list(num_qbits:int,qbit_offset:int=0):
    if num_qbits == 1:
        return [[lowerLeftKernel_hm_list[0], lowerLeftKernel_hm_list[1], [qbit_offset]]]
    return hm_list_tensor([[lowerLeftKernel_hm_list[0], lowerLeftKernel_hm_list[1], [num_qbits-1+qbit_offset]]], get_lowerLeft_hm_list(num_qbits-1,qbit_offset))

@lru_cache
def get_upperRight_hm_list(num_qbits:int,qbit_offset:int=0):
    if num_qbits == 1:
        return [[upperRightKernel_hm_list[0], upperRightKernel_hm_list[1], [qbit_offset]]]
    return hm_list_tensor([[upperRightKernel_hm_list[0], upperRightKernel_hm_list[1], [num_qbits-1+qbit_offset]]], get_upperRight_hm_list(num_qbits-1,qbit_offset))

@lru_cache
def get_laplace1D_hm_list(num_qbits:int):
    if num_qbits == 1:
        return [[laplaceKernel_hm_list[0], laplaceKernel_hm_list[1], [0]]]
    return hm_list_sum(
        hm_list_tensor([[np.zeros(1,np.uint32), np.array([1.0],dtype=np.complex128), [num_qbits-1]]], get_laplace1D_hm_list(num_qbits-1)),
        hm_list_tensor(get_lowerLeft_hm_list(1,num_qbits-1), get_upperRight_hm_list(num_qbits-1)),
        hm_list_tensor(get_upperRight_hm_list(1,num_qbits-1), get_lowerLeft_hm_list(num_qbits-1))
    )

@lru_cache
def get_laplace1D_periodic_hm_list(num_qbits:int):
    return hm_list_sum(
        get_laplace1D_hm_list(num_qbits),
        get_lowerLeft_hm_list(num_qbits),
        get_upperRight_hm_list(num_qbits)
    )

def generateLaplaceHamiltonian1D(num_qbits:int, dx:float,
                                 periodic_bc_flag:bool=False,
                                 homogeneous_flag:bool=False):
    hm_list = []
    if not homogeneous_flag:
        if not periodic_bc_flag:
            hm_list = get_laplace1D_hm_list(num_qbits)
        else:
            hm_list = get_laplace1D_periodic_hm_list(num_qbits)
    else:
        if not periodic_bc_flag:
            hm_list = hm_list_tensor([[upperLeftKernel_hm_list[0], upperLeftKernel_hm_list[1], [num_qbits-1]]],
                                 get_laplace1D_hm_list(num_qbits-1))
        else:
            hm_list = hm_list_tensor([[upperLeftKernel_hm_list[0], upperLeftKernel_hm_list[1], [num_qbits-1]]],
                                 get_laplace1D_periodic_hm_list(num_qbits-1))

    for hm in hm_list:
        for i,amp in enumerate(hm[1]):
            hm[1][i] /= dx*dx
    return Hamiltonian(hm_list, num_qbits)
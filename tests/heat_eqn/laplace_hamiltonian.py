import numpy as np
from numba import njit
from functools import lru_cache
from qnute.hamiltonian import Hamiltonian
from qnute.simulation.parameters import QNUTE_params as Params

def hm_list_adjoint(hm_list):
    return [[hm[0], np.conjugate(hm[1]), hm[2]] for hm in hm_list]

@njit
def hm_pterm_tensor(hm1_ind, hm1_amp, hm2_ind, hm2_amp, len_d2):
    hm_ind = np.zeros(hm1_ind.shape[0]*hm2_ind.shape[0], dtype=np.uint32)
    hm_amp = np.zeros(hm1_ind.shape[0]*hm2_ind.shape[0], dtype=np.complex128)
    k = 0
    for i,p1 in enumerate(hm1_ind):
        a1 = hm1_amp[i]
        for j,p2 in enumerate(hm2_ind):
            a2 = hm2_amp[j]
            hm_ind[k] = (p1*np.left_shift(1, len_d2*2) + p2)
            hm_amp[k] = (a1*a2)
            k += 1
    return hm_ind, hm_amp

def hm_list_tensor(hm_list1, hm_list2):
    hm_list = []
    for hm1 in hm_list1:
        # len_d1 = len(hm1[2])
        for hm2 in hm_list2:
            len_d2 = len(hm2[2])
            hm = [None, 
                  None,
                  hm2[2] + hm1[2]]
            hm[0],hm[1] = hm_pterm_tensor(hm1[0], hm1[1], hm2[0], hm2[1], len_d2)
            hm_list.append(hm)
    return hm_list

@njit
def add_hm_terms(hm1_ind, hm1_amp, hm2_ind, hm2_amp):
    hm_ind = np.zeros(hm1_ind.shape[0] + hm2_ind.shape[0],dtype=np.uint32)
    hm_amp = np.zeros(hm1_ind.shape[0] + hm2_ind.shape[0],dtype=np.complex128)
    
    num_terms = hm1_ind.shape[0]
    hm_ind[0:num_terms] = hm1_ind
    hm_amp[0:num_terms] = hm1_amp
    

    for j,p2 in enumerate(hm2_ind):
        if p2 not in hm_ind[0:num_terms]:
            hm_ind[num_terms] = p2
            hm_amp[num_terms] = hm2_amp[j]
            num_terms += 1
        else:
            k = np.where(hm_ind == p2)[0]
            hm_amp[k] += hm2_amp[j]
    # hm_ind.reshape(num_terms)
    # hm_amp.reshape(num_terms)
    return hm_ind[0:num_terms], hm_amp[0:num_terms]
            
def hm_list_add(hm_list1, hm_list2):
    hm_list = []
    terms_added = [set(),set()] # For hm_list2
    for i,hm1 in enumerate(hm_list1):
        for j,hm2 in enumerate(hm_list2):
            if j not in terms_added[1]:
                if hm1[2] == hm2[2]:
                    hm = [None,None,hm2[2]]
                    hm[0],hm[1] = add_hm_terms(hm1[0], hm1[1], hm2[0], hm2[1])
                    hm_list.append(hm)
                    terms_added[0].add(i)
                    terms_added[1].add(j)
    for i,hm1 in enumerate(hm_list1):
        if i not in terms_added[0]:
            hm_list.append(hm1)
    for j,hm2 in enumerate(hm_list2):
        if j not in terms_added[1]:
            hm_list.append(hm2)
    return hm_list

def hm_list_sum(*args):
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        return hm_list_add(args[0], args[1])
    hm_list = args[0]
    for i in range(1,len(args)):
        hm_list = hm_list_add(hm_list, args[i])
    return hm_list

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

def generateLaplaceHamiltonian1D(num_qbits:int, qubit_map:dict, dx:float,
                                 periodic_bc_flag:bool=False,
                                 homogeneous_flag:bool=False):
    hm_list = []
    invert_map = {value:key for key,value in qubit_map.items()}
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
        for i,coord in enumerate(hm[2]):
            hm[2][i] = invert_map[coord]
        for i,amp in enumerate(hm[1]):
            hm[1][i] /= dx*dx
    return hm_list
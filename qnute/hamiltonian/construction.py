import numpy as np
from functools import lru_cache

from .import hm_list_tensor

lowerLeftKernel_hm_list = [[
    np.array([1,2],dtype=np.uint32),
    np.array([0.5, -0.5j], dtype=np.complex128),
    np.array([0],dtype=np.int32)
]]

upperRightKernel_hm_list = [[
    np.array([1,2],dtype=np.uint32),
    np.array([0.5, 0.5j], dtype=np.complex128),
    np.array([0],dtype=np.int32)
]]

upperLeftKernel_hm_list = [[
    np.array([3,0],dtype=np.uint32),
    np.array([0.5,0.5],dtype=np.complex128),
    np.array([0],dtype=np.int32)
]]

lowerRightKernel_hm_list = [[
    np.array([3,0],dtype=np.uint32),
    np.array([-0.5,0.5],dtype=np.complex128),
    np.array([0],dtype=np.int32)
]]

@lru_cache
def get_lowerLeft_hm_list(num_qbits:int):
    if num_qbits == 1:
        return lowerLeftKernel_hm_list
    return hm_list_tensor(lowerLeftKernel_hm_list, get_lowerLeft_hm_list(num_qbits-1), 1, num_qbits-1)

@lru_cache
def get_lowerRight_hm_list(num_qbits:int):
    if num_qbits == 1:
        return lowerRightKernel_hm_list
    return hm_list_tensor(lowerRightKernel_hm_list, get_lowerRight_hm_list(num_qbits-1), 1, num_qbits-1)

@lru_cache
def get_upperRight_hm_list(num_qbits:int):
    if num_qbits == 1:
        return upperRightKernel_hm_list
    return hm_list_tensor(upperRightKernel_hm_list, get_upperRight_hm_list(num_qbits-1), 1, num_qbits-1)

@lru_cache
def get_upperLeft_hm_list(num_qbits:int):
    if num_qbits == 1:
        return upperLeftKernel_hm_list
    return hm_list_tensor(upperLeftKernel_hm_list, get_upperLeft_hm_list(num_qbits-1), 1, num_qbits-1)

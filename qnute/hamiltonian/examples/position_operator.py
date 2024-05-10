from functools import lru_cache
from .. import Hamiltonian, hm_list_sum, hm_list_tensor, get_identity_hm_list
from ..construction import lowerRightKernel_hm_list, get_upperLeft_hm_list, get_lowerRight_hm_list

@lru_cache
def get_x_hm_list(num_qbits:int):
    if num_qbits == 1:
        return lowerRightKernel_hm_list
    lr_term = hm_list_sum(
        get_x_hm_list(num_qbits-1),
        get_identity_hm_list(num_qbits-1, 2**(num_qbits-1))
    )
    return hm_list_sum(
        hm_list_tensor(get_upperLeft_hm_list(1), get_x_hm_list(num_qbits-1), 1, num_qbits-1),
        hm_list_tensor(get_lowerRight_hm_list(1), lr_term, 1, num_qbits-1)
    )

def generatePositionHamiltonian(num_qbits:int, x0:float, dx:float)->Hamiltonian:
    return (Hamiltonian(get_x_hm_list(num_qbits), num_qbits)*dx) + x0

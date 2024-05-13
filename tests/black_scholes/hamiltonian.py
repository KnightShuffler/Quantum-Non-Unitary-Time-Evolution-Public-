import numpy as np

from qnute.hamiltonian import Hamiltonian, hm_list_tensor
from qnute.hamiltonian.construction import get_lowerLeft_hm_list, get_lowerRight_hm_list
from qnute.hamiltonian.examples.finite_difference.first_derivative import generateFirstDerivativeHamiltonian1D
from qnute.hamiltonian.examples.finite_difference.laplacian import generateLaplaceHamiltonian1D
from qnute.hamiltonian.examples.position_operator import generatePositionHamiltonian

from .import BlackScholesInfo, Basis, BoundaryConditions

def lowerRightHam(num_qbits:int) -> Hamiltonian:
    return Hamiltonian(get_lowerRight_hm_list(num_qbits), num_qbits)
def lowerRight1Ham(num_qbits:int) -> Hamiltonian:
    assert num_qbits >= 2
    return Hamiltonian(hm_list_tensor(get_lowerRight_hm_list(num_qbits-1), 
                                      get_lowerLeft_hm_list(1) ), num_qbits)
def lowerRight2Ham(num_qbits:int) -> Hamiltonian:
    assert num_qbits >=2
    if num_qbits == 2:
        return Hamiltonian(hm_list_tensor(get_lowerLeft_hm_list(1),
                                          get_lowerRight_hm_list(1) ), num_qbits)
    return Hamiltonian(hm_list_tensor(get_lowerRight_hm_list(num_qbits-2),
                                      hm_list_tensor(get_lowerLeft_hm_list(1),
                                                     get_lowerRight_hm_list(1)) ), num_qbits)

def generateBlackScholesHamiltonian(bs_data:BlackScholesInfo,
                                    num_qbits:int
                                    )->Hamiltonian:
    assert num_qbits >= 2, 'Black Scholes Hamiltonian requries at least two qubits!'
    N = 2**num_qbits
    dS = bs_data.Smax / (N-1)

    if bs_data.basis == Basis.S:
        BSHam = (
            (SHam:=generatePositionHamiltonian(num_qbits, 0, dS)) * generateFirstDerivativeHamiltonian1D(num_qbits, dS) * (bs_data.r-bs_data.q) 
            + SHam*SHam*generateLaplaceHamiltonian1D(num_qbits, dS)*((bs_data.sigma**2)/2) 
            + Hamiltonian.Identity(num_qbits)*(-bs_data.r)
            )
        match bs_data.BC:
            case BoundaryConditions.ZERO_AFTER:
                pass
            case BoundaryConditions.ZERO_AT:
                BC_Ham = lowerRightHam(num_qbits) * (bs_data.r + (bs_data.sigma*bs_data.Smax/dS)**2)
                BC_Ham += lowerRight1Ham(num_qbits) * -((bs_data.sigma*bs_data.Smax/dS)**2 / 2 - (bs_data.r-bs_data.q)*bs_data.Smax/(2*dS))
                BSHam += BC_Ham
            case BoundaryConditions.LINEAR:
                BC_Ham = lowerRightHam(num_qbits) * ((bs_data.sigma*bs_data.Smax/dS)**2 + (bs_data.r-bs_data.q)*bs_data.Smax/dS)
                BC_Ham += lowerRight1Ham(num_qbits) * -((bs_data.sigma*bs_data.Smax/dS)**2 / 2 + (bs_data.r-bs_data.q)*bs_data.Smax/(2*dS))
                BSHam += BC_Ham
            case BoundaryConditions.PDE:
                BC_Ham = lowerRightHam(num_qbits) * ((bs_data.sigma*bs_data.Smax/dS)**2 + (bs_data.r-bs_data.q)*bs_data.Smax/dS)
                BC_Ham += lowerRight1Ham(num_qbits) * -((bs_data.sigma*bs_data.Smax/dS)**2 *3/2 + (bs_data.r-bs_data.q)*bs_data.Smax/2*dS)
                BC_Ham += lowerRight2Ham(num_qbits) * (bs_data.sigma*bs_data.Smax/dS)**2
                BSHam += BC_Ham
            case _:
                raise NotImplementedError('These boundary conditions are not yet implemented!')
    else:
        raise NotImplementedError('x-Basis hamiltonian not implemented yet!')
    
    return BSHam
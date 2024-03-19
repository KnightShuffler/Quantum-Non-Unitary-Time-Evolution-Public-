import numpy as np
from numba import njit

import logging

from qnute.hamiltonian.laplacian import generateLaplaceHamiltonian1D
from qnute.simulation.numerical_sim import qnute
from qnute.simulation.parameters import QNUTE_params as Params

@njit
def get_zero_bc_analytical_solution(frequency_amplitudes:np.ndarray[float],L:float, 
                              Nx:int, dx:float, Nt:int, dt:float,
                              alpha:float) -> np.ndarray[float]:
    solutions = np.zeros((Nt+1,Nx), dtype=np.float64)
    x = np.arange(1,Nx+1)*dx
    for ti, t in enumerate(np.arange(Nt+1)*dt):
        for i,amp in enumerate(frequency_amplitudes):
            frequency = i+1
            solutions[ti,:] += amp * np.sin(frequency*np.pi*x/L) * np.exp(-alpha*np.power(frequency*np.pi/L,2)*t)

    return solutions

@njit
def get_periodic_bc_analytical_solution(frequency_amplitudes:np.ndarray[float],L:float, 
                              Nx:int, dx:float, Nt:int, dt:float,
                              alpha:float) -> np.ndarray[float]:
    solutions = np.zeros((Nt+1,Nx),dtype=np.float64)
    x = np.arange(0,Nx)*dx
    for ti, t in enumerate(np.arange(Nt+1)*dt):
        for i,amp in enumerate(frequency_amplitudes):
            if i == 0:
                solutions[ti,:] += amp * np.ones(Nx,dtype=np.float64)
            elif i % 2 == 0:
                frequency = i
                solutions[ti,:] += amp * np.sin(frequency*np.pi*x/L) * np.exp(-alpha*np.power(frequency*np.pi/L,2)*t)
            else:
                frequency = i+1
                solutions[ti,:] += amp * np.cos(frequency*np.pi*x/L) * np.exp(-alpha*np.power(frequency*np.pi/L,2)*t)
    
    return solutions

@njit
def get_zero_bc_frequency_amplitudes(psi:np.ndarray[float], dx:float, L:float
                                     ) -> np.ndarray[float]:
    Nx = psi.shape[0]
    x = np.arange(1,Nx+1)*dx
    frequency_amplitudes = np.zeros(Nx,dtype=np.float64)
    for i in range(Nx):
        nu = i+1
        psi_nu = np.sin(nu*np.pi*x/L)
        # frequency_amplitudes[i] = np.dot(psi_nu, psi) / np.linalg.norm(psi_nu)**2
        frequency_amplitudes[i] = np.dot(psi_nu, psi)

    return frequency_amplitudes * 2.0*dx/L

@njit
def get_periodic_bc_frequency_amplitudes(psi:np.ndarray[float], dx:float, L:float
                                     ) -> np.ndarray[float]:
    Nx = psi.shape[0]
    x = np.arange(Nx)*dx
    frequency_amplitudes = np.zeros(Nx,dtype=np.float64)
    for i in range(Nx):
        if i == 0:
            psi_nu = np.ones(Nx,dtype=np.float64)
        elif i % 2 == 0:
            nu = i
            psi_nu = np.sin(nu*np.pi*x/L)
        else:
            nu = i+1
            psi_nu = np.cos(nu*np.pi*x)
        frequency_amplitudes[i] = np.dot(psi_nu, psi) / np.linalg.norm(psi_nu)**2

    return frequency_amplitudes

def run_1D_heat_eqn_simulation(num_qbits:int, dx:float, T:float, 
                                    dtau:float, alpha:float,
                                    psi0:np.ndarray[float],
                                    periodic_bc_flag:bool,
                                    D_list:list[int]=None,
                                    delta:float=0.1,
                                    reduce_dim_flag:bool=True
                                    ) ->np.ndarray[complex]:
    if D_list == None:
        D_list = list(range(2,num_qbits+2,2))
    
    Nx = 2**num_qbits

    dt = dtau*dx*dx
    Nt = np.int32(np.ceil(T/dt))

    solutions = np.zeros((len(D_list), Nt+1, Nx), dtype=np.complex128)

    H = generateLaplaceHamiltonian1D(num_qbits, periodic_bc_flag=periodic_bc_flag)
    H.multiply_scalar(alpha)

    c0 = np.linalg.norm(psi0)
    psi0 /= c0

    qubit_map = {(i,):i for i in range(num_qbits)}

    params = Params(H, 1, num_qbits, qubit_map)

    for Di,D in enumerate(D_list):
        if D < num_qbits:
            u_domains = [list(range(i,i+D)) for i in range(num_qbits-D+1)]
        else:
            u_domains = [list(range(num_qbits))]
        
        params.load_hamiltonian_params(D, u_domains, reduce_dim_flag, True)
        params.set_run_params(dtau, delta, Nt, 0, None, init_sv=psi0,trotter_flag=True)


        out = qnute(params, log_frequency=100, c0=c0)
        solutions[Di,:,:] = out.svs
        for ti in range(Nt+1):
            solutions[Di,ti,:] *= np.prod(out.c_list[0:ti+1])

    return solutions

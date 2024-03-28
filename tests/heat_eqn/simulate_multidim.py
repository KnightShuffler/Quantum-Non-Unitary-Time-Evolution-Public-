import numpy as np
from numba import njit

import logging 

from qnute.hamiltonian.laplacian import generateLaplacianHamiltonianMultiDim
from qnute.simulation.numerical_sim import qnute
from qnute.simulation.parameters import QNUTE_params as Params

from .input_handler import ExperimentInput

from . import heat_logger

def get_fourier_eigenstates(num_qbits:np.ndarray[int],
                            periodic_bc_flags:np.ndarray[bool]):
    ndim = num_qbits.shape[0]
    Nx = 2**num_qbits
    num_freq = 2**(np.sum(num_qbits))
    

    state_counter = np.zeros(ndim,dtype=np.int32)

    for i in range(num_freq):
        # Get eigenstate:
        eigenstate = np.array([1],dtype=np.float64)
        normalization = 1.0
        frequencies = np.zeros(ndim,dtype=np.int32)

        for j in range(ndim):
            if not periodic_bc_flags[j]:
                x = np.arange(1,Nx[j]+1)
                nu = state_counter[j] + 1
                eigenstate = np.kron(np.sin(nu*np.pi*x/(Nx[j]+1)), eigenstate)
                normalization *= (Nx[j]+1)/2.0
            else:
                x = np.arange(Nx[j])
                if state_counter[j] == 0:
                    nu = 0
                    eigenstate = np.kron(np.ones(Nx[j]), eigenstate)
                    normalization *= Nx[j]
                elif state_counter[j] % 2 == 0:
                    nu = state_counter[j]
                    eigenstate = np.kron(np.sin(nu*np.pi*x/Nx[j]), eigenstate)
                    normalization *= Nx[j]/2.0
                else:
                    nu = state_counter[j] + 1
                    eigenstate = np.kron(np.cos(nu*np.pi*x/Nx[j]), eigenstate)
                    if nu != Nx[j]:
                        normalization *= Nx[j]/2.0
                    else:
                        normalization *= Nx[j]
            frequencies[j] = nu
            
        yield eigenstate,normalization,frequencies
        
        # update state counter
        for j in range(ndim):
            state_counter[j] += 1
            if state_counter[j] != Nx[j]:
                break
            else:
                state_counter[j] = 0

def get_fourier_amplitudes(psi, num_qbits, periodic_bc_flags):
    amplitudes = np.zeros(psi.shape[0], dtype=np.float64)
                        #   dtype=np.dtype([
                        #       ('amplitude', np.float64), 
                        #       ('frequencies', np.int32, (num_qbits.shape[0],))
                        #       ]))
    for i, (state,factor,freqs) in enumerate(get_fourier_eigenstates(num_qbits, periodic_bc_flags)):
        # amplitudes[i]['amplitude'] = np.dot(state, psi) / factor
        # amplitudes[i]['frequencies'] = freqs
        amplitudes[i] = np.dot(state, psi) / factor
    return amplitudes

def get_analytical_solution(fourier_amplitudes:np.ndarray[float],
                            num_qbits:np.ndarray[int],
                            periodic_bc_flags:np.ndarray[bool],
                            dx:np.ndarray[float],
                            L:np.ndarray[float],
                            Nt:int,
                            dt:float,
                            alpha:float
                            )->np.ndarray[float]:
    N = fourier_amplitudes.shape[0]
    ndims = num_qbits.shape[0]
    analytical_sols = np.zeros((Nt+1,N), dtype=np.float64)
    for i in range(Nt+1):
        t = i*dt
        for j, (state,factor,freqs) in enumerate(get_fourier_eigenstates(num_qbits, periodic_bc_flags)):
            k = 0.0
            for dim in range(ndims):
                k += (freqs[dim]*np.pi/L[dim])**2
            k *= alpha
            analytical_sols[i,:] += fourier_amplitudes[j] * np.exp(-k*t) * state        

    return analytical_sols


def run_heat_eqn_simulation(expt:ExperimentInput, independent_cover_flag:bool=True) -> np.ndarray[float]:
    min_dx = np.min(expt.dx)
    dt = expt.dtau * min_dx**2
    Nt = np.int32(np.ceil(expt.T / dt))
    total_qbits = np.sum(expt.num_qbits)
    N = 2**(total_qbits)

    solutions = np.zeros((numD:=expt.D_list.shape[0], Nt+1, N), dtype=np.complex128)

    H = generateLaplacianHamiltonianMultiDim(expt.num_qbits, expt.dx, expt.periodic_bc_flag)
    H.multiply_scalar(expt.alpha)

    c0 = np.linalg.norm(expt.f0)
    psi0 = expt.f0 / c0

    qubit_map = {(i,):i for i in range(np.sum(expt.num_qbits))}

    params = Params(H, 1, total_qbits, qubit_map)

    for Di,D in enumerate(expt.D_list):
        if not independent_cover_flag:
            if D < total_qbits:
                u_domains = [list(range(i,i+D)) for i in range(total_qbits-D+1)]
            else:
                u_domains = [list(range(total_qbits))]
        else:
            u_domains = []
            for j in range(expt.num_space_dims):
                start = expt.num_qbits[j-1] if j > 0 else 0
                if D < expt.num_qbits[j]:
                    for i in range(expt.num_qbits[j]-D+1):
                        u_domains.append(list(range(i+start,i+start+D)))
                else:
                    u_domains.append(list(range(start, start + expt.num_qbits[j])))

        params.load_hamiltonian_params(D, u_domains, True, True)
        params.set_run_params(expt.dtau, 0.1, Nt, 0, None, init_sv=psi0, trotter_flag=True)

        out = qnute(params, log_frequency=100, c0=c0)
        solutions[Di,:,:] = out.svs
        for ti in range(Nt+1):
            solutions[Di,ti,:] *= np.prod(out.c_list[0:ti+1])

    return solutions.real

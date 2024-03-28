import numpy as np
from numba import njit

import logging 

from qnute.hamiltonian.laplacian import generateLaplaceHamiltonian1D
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


def run_heat_eqn_simulation(expt:ExperimentInput) -> np.ndarray[complex]:
    pass
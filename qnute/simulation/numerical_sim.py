import numpy as np
from numba import njit

import logging
import time

from qnute.helpers import exp_mat_psi
from qnute.helpers.pauli import pauli_string_prod
from qnute.helpers.pauli import get_pauli_prod_matrix

from .parameters import QNUTE_params as Params
from .output import QNUTE_output as Output

pauli_pair_dtype = np.dtype([('pauli_id',np.uint32), ('value',np.float64)])

@njit
def get_theoretical_evolution(H_mat:np.array, psi0: np.array, dt:float, N:int):
    '''Numerically calculates the theoretical time evolution exp(Ht)|psi_0> with Taylor
    series'''
    times = np.arange(0,N+1,1)*dt
    svs = np.zeros((N+1, psi0.shape[0]),dtype='c16')
    for (i,t) in enumerate(times):
        svs[i] = exp_mat_psi(t*H_mat, psi0)

        svs[i] /= np.linalg.norm(svs[i])
    return svs

def tomography(params:Params, psi0:np.array, 
               a_list:list, term:int) -> dict:
    sigma_expectation = {
        'c': np.zeros(params.h_measurements[term].shape[0],dtype=pauli_pair_dtype), # acts on h_domains[m]
        'S': np.zeros(params.u_measurements[term].shape[0],dtype=pauli_pair_dtype), # acts on u_domains[m]
        'b': np.zeros(params.mix_measurements[term].shape[0],dtype=pauli_pair_dtype)  # acts on mix_domains[m]
    }
    psi = propagate(params, psi0, a_list)

    # If the unitary domain radius >= the hamiltonian term domain radius, all the
    # measurement operators of S cover all the measurement we need to take
    for i,p in enumerate(params.u_measurements[term]):
        sigma_expectation['S'][i] = (p,pauli_expectation(psi, p, params.nbits, params.num_shots))
    # Otherwise, we need to account for all the different measurement domains
    for i,p in enumerate(params.h_measurements[term]):
        sigma_expectation['c'][i] = (p, pauli_expectation(psi, p, params.nbits, params.num_shots))
    for i,p in enumerate(params.mix_measurements[term]):
        sigma_expectation['b'][i] = (p, pauli_expectation(psi, p, params.nbits, params.num_shots))
        
    return sigma_expectation

@njit
def pauli_expectation(psi:np.array, p:int, nbits:int, num_shots:int = 0) -> np.float64:
    '''Returns the expectation value of the Pauli measurement indexed by p, acting
    on the qubits whose indices are given in qbits'''
    if p == 0:
        return 1.0
    # Theoretical Expectation
    if num_shots == 0:
        p_mat = get_pauli_prod_matrix(p, nbits)
        return np.real(np.vdot(psi, np.dot(p_mat, psi)))
    
@njit
def construct_c(pterms:np.array, c_expectations:np.array,  scale:float, dt:float) -> np.float64:
    c = 1.0
    for j,pterm in enumerate(pterms):
        c += 2*scale * dt * np.real(pterm['amplitude']) * c_expectations[j]['value']
    return np.sqrt(c)

@njit
def construct_S(u_measurements, S_expectations, nbits):
    nops = u_measurements.shape[0]
    S = np.zeros((nops,nops), dtype=np.complex128)
    for i,p_I in enumerate(u_measurements):
        for j,p_J in enumerate(u_measurements):
            p_,c_ = pauli_string_prod(p_I, p_J, nbits)
            S[i,j] = S_expectations['value'][np.where(S_expectations['pauli_id']==p_)][0] * c_
    return S

@njit
def construct_b(pterms, u_measurements, b_expectations, c, nbits, scale):
    b = np.zeros(u_measurements.shape[0], dtype=np.float64)
    for i,p_I in enumerate(u_measurements):
        for j,p_J in enumerate(pterms):
            p_,c_ = pauli_string_prod(p_I, p_J['pauli_id'], nbits)
            b[i] += scale * np.imag(pterms[j]['amplitude'] * c_) * b_expectations['value'][np.where(b_expectations['pauli_id']==p_)][0]
    b *= -(2.0 / c)
    return b

@njit
def solve_for_a_list(S, b, delta, u_measurements):
    dalpha = np.eye(u_measurements.shape[0]) * delta
    a = np.real(np.linalg.lstsq(2*np.real(S) + dalpha, b, rcond=-1)[0])
    
    a_list_term = np.zeros(a.shape[0], dtype=pauli_pair_dtype)
    for i,p in enumerate(u_measurements):
        a_list_term[i]['pauli_id'] = p
        a_list_term[i]['value'] = a[i]
    return a_list_term

def update_alist(params:Params, sigma_expectation:dict, 
             term:int, psi0:np.array, 
             scale:float):
    # hm = params.QNUTE_H.hm_list[term]
    pterms = params.QNUTE_H.get_hm_pterms(term)
    num_terms = pterms.shape[0]
    u_domain = params.u_domains[term]
    ndomain = len(u_domain)

    # Load c
    c = construct_c(pterms, sigma_expectation['c'], scale, params.dt)
    
    # Load S
    S = construct_S(params.u_measurements[term], sigma_expectation['S'], params.nbits)
    
    # Load b
    b = construct_b(pterms, params.u_measurements[term], sigma_expectation['b'], c, params.nbits, scale)

    a_list_term = solve_for_a_list(S,b,params.delta,params.u_measurements[term])
    
    return a_list_term, c
   
def qnute(params:Params, log_frequency:int = 10, c0:float=1.0) -> Output:
    output = Output(params)
    output.svs[0,:] = params.init_sv.data
    output.c_list.append(c0)

    logging.info('Performing initial measurements...')
    for m in params.objective_measurements:
        output.measurements[m[0]][0] = pauli_expectation(params, params.init_sv, m[1],m[2])
    logging.info('Starting QNUTE Iterations:')

    for i in range(1, params.N+1):
        if i % log_frequency == 0 or i == params.N or i == 1:
            logging.info(f'    Iteration {i:03d}')
        
        t0 = time.monotonic()

        qnute_step(params, output, i)

        t1 = time.monotonic()
        output.times[i-1] = t1 - t0
        if i % log_frequency == 0 or i == params.N or i == 1:
            logging.info(f'      Finished in {(t1-t0):0.2f} seconds.')
    
    return output

def qnute_step(params:Params, output:Output, step:int):
    a_list = []
    c_list = []

    psi0 = output.svs[step - 1]
    H = params.QNUTE_H

    for term in range(H.num_terms):
        if len(params.u_domains[term]) == 0:
            continue
        a = np.array([(0,10.0)],dtype=pauli_pair_dtype)
        while np.linalg.norm(a['value']) > 5.0:
            sigma_expectation = tomography(params, psi0, a_list, term)
            (a, c) = update_alist(params, sigma_expectation, term, propagate(params, psi0, a_list), 1.0)
        
        a_list.append(a)
        c_list.append(c)
    output.a_list += a_list
    output.c_list.append(np.prod(c_list))
    output.svs[step,:] = propagate(params, psi0, a_list)#.data

    for m in params.objective_measurements:
        output.measurements[m[0]][step] = pauli_expectation(params, output.svs[step], m[1], m[2])

def propagate(params:Params, psi0:np.array, 
              a_list:list) -> np.array:
    '''Applies the rotations on state psi0 according to a_list.
    a_list = [ [ [angle list], [qubit_indices], [reduced_dimension?(False for rn)] ],
                   ...
               [ [], [], [] ], ]'''
    psi = psi0.copy()
    for (t,a_term_list) in enumerate(a_list):
        psi = prop_a_term(psi, a_term_list, params.nbits, params.dt, params.trotter_flag, params.taylor_truncate_a)
        # A = np.zeros((2**params.nbits, 2**params.nbits), dtype=np.complex128)
        # for (i,a) in enumerate(a_term_list):
        #     p_mat = get_pauli_prod_matrix(a['pauli_id'], params.nbits)
        #     A += a['value'] * p_mat
        #     if params.trotter_flag:
        #         psi = exp_mat_psi(-1j * a['value'] * params.dt * p_mat, psi, truncate=params.taylor_truncate_a)
        # if not params.trotter_flag:
        #     psi = exp_mat_psi(-1j*params.dt*A, psi, truncate=params.taylor_truncate_a)
    psi /= np.linalg.norm(psi)
    return psi

@njit
def prop_a_term(psi0:np.array, a_term_list:np.array, nbits:int, dt:float, trotter_flag:bool, taylor_truncate_a:int) -> np.array:
    psi = psi0.copy()
    A = np.zeros((2**nbits, 2**nbits), dtype=np.complex128)
    for (i,a) in enumerate(a_term_list):
        p_mat = get_pauli_prod_matrix(a['pauli_id'], nbits)
        A += a['value'] * p_mat
        if trotter_flag:
            psi = exp_mat_psi(-1j * a['value'] * dt * p_mat, psi, truncate=taylor_truncate_a)
    if not trotter_flag:
        psi = exp_mat_psi(-1j*dt*A, psi, truncate=taylor_truncate_a)
    return psi
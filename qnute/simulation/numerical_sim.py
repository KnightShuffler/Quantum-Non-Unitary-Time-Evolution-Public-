import numpy as np

import logging
import time

from qnute.helpers import exp_mat_psi, int_to_base
from qnute.helpers.pauli import pauli_string_prod, pauli_index_to_dict
from qnute.helpers.pauli import get_full_pauli_product_matrix, get_pauli_eigenspace
from qnute.helpers.circuit import get_qc_bases_from_pauli_dict

from .parameters import QNUTE_params as Params
from .output import QNUTE_output as Output

def get_theoretical_evolution(H_mat:np.array, psi0: np.array, dt:float, N:int):
    '''Numerically calculates the theoretical time evolution exp(-iHt)|psi_0> with Taylor
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
        'c': {}, # acts on h_domains[m]
        'S': {}, # acts on u_domains[m]
        'b': {}  # acts on mix_domains[m]
    }
    psi = propagate(params, psi0, a_list)

    # If the unitary domain radius >= the hamiltonian term domain radius, all the
    # measurement operators of S cover all the measurement we need to take
    for p in params.u_measurements[term]:
        sigma_expectation['S'][p] = pauli_expectation(params, psi, p, params.u_domains[term])
    # Otherwise, we need to account for all the different measurement domains
    if params.small_u_domain_flags[term]:
        for p in params.h_measurements[term]:
            sigma_expectation['c'][p] = pauli_expectation(params, psi, p, params.h_domains[term])
        for p in params.mix_measurements[term]:
            sigma_expectation['b'][p] = pauli_expectation(params, psi, p, params.mix_domains[term])
        
    return sigma_expectation

def update_alist(params:Params, sigma_expectation:dict, 
             a_list:list, term:int, psi0:np.array, 
             scale:float):
    hm = params.H.hm_list[term]
    num_terms = len(hm[0])
    u_domain = params.u_domains[term]
    ndomain = len(u_domain)

    def full_pauli_index(p, d1, d2):
        index = 0
        for (i,qbit) in enumerate(d1):
            gate = p % 4
            # find the index of qbit in d2
            if qbit not in d2:
                raise ValueError(f'Element {qbit} not found in domain {d2}')
            j = d2.index(qbit)

            index += gate * 4**j
            p = p//4
        return index
    
    # Load c
    if params.taylor_norm_flag:
        h_mat = params.H.get_term_submatrix(term)
        psi_prime = exp_mat_psi(params.dt*h_mat, psi0.data, truncate=params.taylor_truncate_h)
        c = np.linalg.norm(psi_prime)
    else:
        c = 1.0
        for j in range(num_terms):
            if params.small_u_domain_flags[term]:
                key1 = 'c'
                key2 = params.h_measurements[term][j]
            else:
                key1 = 'S'
                key2 = full_pauli_index(params.h_measurements[term][j], 
                                        params.h_domains[term], 
                                        params.u_domains[term])
            c += 2*scale * params.dt * np.real(hm[1][j]) * sigma_expectation[key1][key2]
        c = np.sqrt(c)
    
    # Load S
    if params.H.real_term_flags[term] and params.reduce_dimension_flag:
        ops = params.odd_y_strings[ndomain]
    else:
        ops = list(range(4**ndomain))
    nops = len(ops)
    
    S = np.zeros((nops,nops), dtype=complex)
    for i in range(nops):
        I = ops[i]
        for j in range(0,i):
            J = ops[j]
            p_,c_ = pauli_string_prod(I, J, ndomain)
            S[i,j] = sigma_expectation['S'][p_] * c_
            # S is Hermitian, so we know the upper triangle
            S[j,i] = S[i,j].conjugate()
        # The diagonal is full of 1s: <psi|I|psi>
        S[i,i] = 1.0
    
    # Load b
    b = np.zeros(nops, dtype=float)
    for i in range(nops):
        if params.small_u_domain_flags[term]:
            key1 = 'b'
            I = full_pauli_index(ops[i], 
                                u_domain, 
                                params.mix_domains[term])
        else:
            key1 = 'S'
            I = ops[i]
        for j in range(num_terms):
            if params.small_u_domain_flags[term]:
                J = full_pauli_index(params.h_measurements[term][j], 
                                    params.h_domains[term], 
                                    params.mix_domains[term])
            else:
                J = full_pauli_index(params.h_measurements[term][j],
                                    params.h_domains[term],
                                    u_domain)

            if params.small_u_domain_flags[term]:
                p_,c_ = pauli_string_prod(I, J, len(params.mix_domains[term]))
            else:
                p_,c_ = pauli_string_prod(I, J, ndomain)

            b[i] += scale * np.imag(hm[1][j] * c_) * sigma_expectation[key1][p_]
    b = -(2.0 / c) * b

    #Regularizer
    dalpha = np.eye(nops) * params.delta
    
    a = np.real(np.linalg.lstsq(2*np.real(S) + dalpha, b, rcond=-1)[0])
    return a, c

def pauli_expectation(params:Params, psi:np.array, p:int, qbits:list):
    '''Returns the expectation value of the Pauli measurement indexed by p, acting
    on the qubits whose indices are given in qbits'''
    def theoretical_expectation(bases, active):
        '''Returns the theoretical expectation <psi|P|psi>'''
        p_mat = get_full_pauli_product_matrix([bases[q] for q in active], active, params.nbits)
        return np.real(np.vdot(psi.data, p_mat @ psi.data))
    
    def statistical_expectation(bases, active):
        '''Simulates measurements according to the measurement probability of psi'''
        pos_eigenspace = get_pauli_eigenspace([bases[q] for q in active], active, params.nbits, 1.0)
        pos_prob = np.clip(np.sum(np.abs(pos_eigenspace.conj() @ psi.data)**2), 0.0, 1.0)
        pos_meas = np.random.binomial(params.num_shots, pos_prob)
        return 2.0*pos_meas/params.num_shots - 1.0
    
    if p == 0:
        return 1.0
    
    p_dict = pauli_index_to_dict(p, qbits)
    bases = get_qc_bases_from_pauli_dict(p_dict, params.H.map)
    active = list(bases.keys())
    
    if params.num_shots > 0:
        return statistical_expectation(bases, active)
    else:
        return theoretical_expectation(bases, active)
   
def qnute(params:Params, log_frequency:int = 10) -> Output:
    output = Output(params)
    output.svs[0,:] = params.init_sv.data

    logging.debug('Performing initial measurements...')
    for m in params.objective_measurements:
        output.measurements[m[0]][0] = pauli_expectation(params, params.init_sv, m[1],m[2])
    logging.debug('Starting QNUTE Iterations:')

    for i in range(1, params.N+1):
        if i % log_frequency == 0 or i == params.N or i == 1:
            logging.debug(f'    Iteration {i:03d}')
        
        t0 = time.monotonic()

        qnute_step(params, output, i)

        t1 = time.monotonic()
        output.times[i-1] = t1 - t0
        if i % log_frequency == 0 or i == params.N or i == 1:
            logging.debug(f'      Finished in {(t1-t0):0.2f} seconds.')
    
    return output

def qnute_step(params:Params, output:Output, step:int):
    a_list = []
    c_list = []

    psi0 = output.svs[step - 1]
    H = params.H

    for term in range(H.num_terms):
        if len(params.u_domains[term]) == 0:
            continue
        a = 10.0
        while np.linalg.norm(a) > 5.0:
            sigma_expectation = tomography(params, psi0, a_list, term)
            (a, c) = update_alist(params, sigma_expectation, a_list, term, propagate(params, psi0, a_list), 1.0)
        
        a_list.append([list(a), params.u_domains[term], params.H.real_term_flags[term] and params.reduce_dimension_flag])
        c_list.append(c)
    output.a_list += a_list
    output.c_list += c_list
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
    for (t,a) in enumerate(a_list):
        active = [params.H.map[k if isinstance(k, tuple) else (k,)] for k in a[1]]
        nactive = len(active)
        ops = params.odd_y_strings[nactive] if a[2] else list(range(4**nactive))
        A = np.zeros((2**params.nbits, 2**params.nbits), dtype='c16')
        for (i,op) in enumerate(ops):
            p_mat = get_full_pauli_product_matrix(int_to_base(op, 4, nactive), active, params.nbits)
            A += a[0][i] * p_mat
            if params.trotter_flag:
                psi = exp_mat_psi(-1j * a[0][i] * params.dt * p_mat, psi, truncate=params.taylor_truncate_a)
        if not params.trotter_flag:
            psi = exp_mat_psi(-1j*params.dt*A, psi, truncate=params.taylor_truncate_a)
    psi /= np.linalg.norm(psi)
    return psi

# Functional version
def _propagate(params, psi0, a_list):
    """Applies the rotations on state psi0 according to a_list."""
    def get_active_indices(qubit_indices):
        logging.debug(f'qubit_indices: {qubit_indices}')
        return [params.H.map[k if isinstance(k, tuple) else (k,)] for k in qubit_indices]

    def calculate_ops(a):
        active = get_active_indices(a[1])
        nactive = len(active)
        return params.odd_y_strings[nactive] if a[2] else list(range(4**nactive))

    def apply_rotation(psi, angle, op, truncate):
        p_mat = get_full_pauli_product_matrix(int_to_base(op, 4, len(active)), active, params.nbits)
        return exp_mat_psi(-1j * angle * params.dt * p_mat, psi, truncate=truncate)

    def apply_rotation_no_trotter(psi, angle, A, truncate):
        return exp_mat_psi(-1j * angle * params.dt * A, psi, truncate=truncate)

    psi = psi0.copy()

    for (t, a) in enumerate(a_list):
        active = get_active_indices(a[1])
        ops = calculate_ops(a)
        A = np.zeros((2**params.nbits, 2**params.nbits), dtype='c16')
        for (i, op) in enumerate(ops):
            A += a[0][i] * get_full_pauli_product_matrix(int_to_base(op, 4, len(active)), active, params.nbits)
            if params.trotter_flag:
                psi = apply_rotation(psi, a[0][i], op, params.taylor_truncate_a)
        if not params.trotter_flag:
            psi = apply_rotation_no_trotter(psi, params.dt, A, params.taylor_truncate_a)

    return psi

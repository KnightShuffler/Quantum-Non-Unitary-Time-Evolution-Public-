import numpy as np
from qiskit import QuantumCircuit
from qiskit import Aer
from qiskit.quantum_info import Statevector

from qnute.hamiltonian import Hamiltonian
from qnute.helpers import *
from qnute.helpers.pauli import *
from qnute.helpers.circuit import *

from .parameters import QNUTE_params as Params
from .output import QNUTE_output as Output

sv_sim = Aer.get_backend('statevector_simulator')

import time

def evolve_statevector(params: Params, qc, psi):
    '''
    Evolves the statevector psi through the circuit qc and returns the statevector
    '''
    circ = QuantumCircuit(params.nbits)
    circ.initialize(psi)
    circ.compose(qc, inplace=True)
    result = execute(circ, params.backend).result()
    return result.get_statevector(circ)

def propagate(params: Params, psi0, a_list):
    qc = QuantumCircuit(params.nbits)
    qc.initialize(psi0, list(range(params.nbits)))
    for t in range(len(a_list)):
        domain = a_list[t][1]
        ndomain = len(domain)
        ops = params.odd_y_strings[ndomain] if a_list[t][2] else list(range(4**ndomain))
        for i in range(len(ops)):
            # Skip index 0 for the non-real hamiltonian term since that's just a global phase
            if not a_list[t][2] and i == 0:
                continue

            angle = a_list[t][0][i] * 2.0 * params.dt
            if np.abs(angle) > TOLERANCE:
                p_dict = pauli_index_to_dict(ops[i], domain)
                pauli_string_exp(qc, p_dict, params.H.qubit_map, angle)
    return evolve_statevector(params, qc, psi0)

def pauli_expectation(params: Params, psi, p, qbits):
    '''
    returns the expectation <psi|P|psi> where P is the pauli string acting on qbits, 
    described using a Pauli string dictionary
    '''
    if p == 0:
        return 1.0
    p_dict = pauli_index_to_dict(p, qbits)
    assert params.num_shots > 0, 'params.num_shots must be > 0 when using QuantumCircuits'
    qc = QuantumCircuit(params.nbits,params.nbits)
    qc.initialize(psi)
    return measure(qc, p_dict, params.H.qubit_map, params.backend, num_shots=params.num_shots)[0]

def tomography(params: Params, psi0, a_list, term):
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

def update_alist(params: Params, sigma_expectation, a_list, term, psi0, scale):
    hm = params.H.hm_list[term]
    num_terms = len(hm[0])
    u_domain = params.u_domains[term]
    ndomain = len(u_domain)

    def full_pauli_index(p, d1, d2):
        index = 0
        for i in range(len(d1)):
            gate = p % 4
            
            qbit = d1[i]
            # find the index of qbit in d2
            j=0
            while d2[j] != qbit:
                j += 1
                if j == len(d2):
                    raise ValueError('Element {} not found in domain {}'.format(d1[i],d2) )

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
    
    # a_list.append([list(a), u_domain, params.H.real_term_flags[term] and params.reduce_dimension_flag])
    # return S,b,c
    return a, c

def qnute_step(params: Params, output:Output, step):
    a_list = []
    # S_list = []
    # b_list = []
    c_list = []
    
    psi0 = output.svs[step-1]
    H = params.H
    
    for term in range(H.num_terms):
        # Ignore terms of empty unitary domains
        if len(params.u_domains[term]) == 0:
            continue
        a = 10.0
        while np.linalg.norm(a) > 5.0:
            sigma_expectation = tomography(params, psi0, a_list, term)
            # S,b,
            a, c = update_alist(params, sigma_expectation, a_list, term, propagate(params, psi0, a_list), 1.0)

        # output.exp.append(sigma_expectation)
        a_list.append([list(a), params.u_domains[term], params.H.real_term_flags[term] and params.reduce_dimension_flag])
        # S_list.append(S)
        # b_list.append(b)
        c_list.append(c)
    
    output.a_list += a_list
    # output.S_list += S_list
    # output.b_list += b_list
    output.c_list += c_list
    output.svs[step,:] = propagate(params, psi0, a_list).data

    for m in params.objective_measurements:
        output.measurements[m[0]][step] = pauli_expectation(params, Statevector(output.svs[step]), m[1], m[2])

def qnute(params:Params, log_to_console:bool=True, log_frequency:int=10):
    output = Output(params)
    output.svs[0,:] = params.init_sv.data

    if log_to_console: print('Performing initial measurements:...',end=' ',flush=True)
    for m in params.objective_measurements:
        output.measurements[m[0]][0] = pauli_expectation(params, params.init_sv, m[1],m[2])
    if log_to_console: print('Done')
    
    if log_to_console: print('Starting QNUTE Iterations:')
    
    for i in range(1, params.N+1):
        if log_to_console:
            if i % log_frequency == 0 or i == params.N or i == 1:
                print('Iteration {}...'.format(i),end=' ',flush=True)
        
        t0 = time.time()
        
        qnute_step(params, output, i)
                
        t1 = time.time()
        duration = t1-t0
        output.times[i-1] = duration
        
        if log_to_console: 
            if i % log_frequency == 0 or i == params.N or i == 1:
                print('Done -- Iteration time = {:0.2f} {}'.format(duration if duration < 60 else duration/60, 'seconds' if duration < 60 else 'minutes'))
    
    return output

import numpy as np
CP_IMPORT_FLAG = True
try:
    import cupy as cp
except ImportError as e:
    print('cupy failed to import')
    CP_IMPORT_FLAG = False
import time

from helpers import *
from qnute_params import QNUTE_params, DRIFT_NONE, DRIFT_A, DRIFT_THETA_2PI, DRIFT_THETA_PI_PI

from qiskit import QuantumCircuit, execute
from qiskit.quantum_info import Statevector

# Note: These methods implicitly assumes that the backend is the Statevector simulator

def evolve_statevector(params: QNUTE_params, qc, psi):
    '''
    Evolves the statevector psi through the circuit qc and returns the statevector
    '''
    circ = QuantumCircuit(params.nbits)
    circ.initialize(psi)
    circ = circ + qc
    result = execute(circ, params.backend).result()
    return result.get_statevector(circ)

def pauli_expectation(params: QNUTE_params, psi, p, qbits, qubit_map):
    '''
    returns the theoretical expectation <psi|P|psi> where P is the pauli string acting on qbits, 
    described using a Pauli string dictionary
    '''
    p_dict = pauli_index_to_dict(p, qbits)

    bases = get_qc_bases_from_pauli_dict(p_dict, qubit_map)
    active = list(bases.keys())
    qc = QuantumCircuit(params.nbits)
    for i in active:
        if bases[i] == 1:
            qc.x(i)
        elif bases[i] == 2:
            qc.y(i)
        elif bases[i] == 3:
            qc.z(i)
    
    phi = evolve_statevector(params, qc, psi)
    
    return np.real(np.vdot(psi.data, phi.data))

def measure_energy(params: QNUTE_params, psi):
    '''
    returns the mean energy <psi|H|psi>
    '''
    E = 0.0
    for m in range(params.H.num_terms):
        hm = params.H.hm_list[m]
        for j in range(len(hm[0])):
            E += pauli_expectation(params, psi, params.h_measurements[m][j], params.h_domains[m], params.H.map) * hm[1][j]
    return E

def propagate(params: QNUTE_params, psi0, alist):
    qc = QuantumCircuit(params.nbits)
    qc.initialize(psi0, list(range(params.nbits)))
    for t in range(len(alist)):
        domain = alist[t][1]
        ndomain = len(domain)
        ops = params.odd_y_strings[ndomain] if alist[t][2] else list(range(4**ndomain))
        for i in range(len(ops)):
            # Skip index 0 for the non-real hamiltonian term since that's just a global phase
            if not alist[t][2] and i == 0:
                continue

            angle = alist[t][0][i]
            if np.abs(angle) > TOLERANCE:
                p_dict = pauli_index_to_dict(ops[i], domain)
                pauli_string_exp(qc, p_dict, params.H.map, angle)
    
    return evolve_statevector(params, qc, psi0)

def tomography(params: QNUTE_params, psi0, alist, term):
    sigma_expectation = { 
        'c': {}, # acts on h_domains[m]
        'S': {}, # acts on u_domains[m]
        'b': {}  # acts on mix_domains[m]
    }

    psi = propagate(params, psi0, alist)

    # If the unitary domain radius >= the hamiltonian term domain radius, all the
    # measurement operators of S cover all the measurement we need to take
    for p in params.u_measurements[term]:
        sigma_expectation['S'][p] = pauli_expectation(params, psi, p, params.u_domains[term], params.H.map)
    # Otherwise, we need to account for all the different measurement domains
    if params.small_u_domain_flags[term]:
        for p in params.h_measurements[term]:
            sigma_expectation['c'][p] = pauli_expectation(params, psi, p, params.h_domains[term], params.H.map)
        for p in params.mix_measurements[term]:
            sigma_expectation['b'][p] = pauli_expectation(params, psi, p, params.mix_domains[term], params.H.map)
        
    return sigma_expectation

def update_alist(params: QNUTE_params, sigma_expectation, alist, term, scale):
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
    b = np.zeros(nops, dtype=complex)
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
            if not params.small_u_domain_flags[term]:
                J = full_pauli_index(params.h_measurements[term][j], 
                                    params.h_domains[term], 
                                    params.mix_domains[term])
            else:
                J = full_pauli_index(params.h_measurements[j],
                                    params.h_domains[term],
                                    u_domain)

            if not params.small_u_domain_flags[term]:
                p_,c_ = pauli_string_prod(I, J, len(params.mix_domains[term]))
            else:
                p_,c_ = pauli_string_prod(I, J, ndomain)

            b[i] += scale * np.imag(hm[1][j] * c_) * sigma_expectation[key1][p_]
    b = -(2.0 / np.sqrt(c)) * b

    #Regularizer
    dalpha = np.eye(nops) * params.delta
    
    if not params.gpu_calc_flag:
        a = np.real(np.linalg.lstsq(2*np.real(S) + dalpha, b, rcond=-1)[0])
    else:
        S_ = cp.asarray(np.real(S) + dalpha)
        b_ = cp.asarray(b)
        a = np.real(cp.linalg.lstsq(2*S_,b_,rcond=-1)[0].get())
    
    # Update alist depending on the drift type of the run
    if params.drift_type == DRIFT_A:
        theta_coeffs = 2.0 * params.dt* sample_from_a(a)
    else:
        thetas = 2.0 * params.dt * a
        
        if params.drift_type == DRIFT_THETA_2PI:
            # Fix the angles between [0,2pi)
            thetas = np.mod(thetas, 2.0*np.pi)
            # Sample from this
            theta_coeffs = sample_from_a(thetas)
        elif params.drift_type == DRIFT_THETA_PI_PI:
            # Fix the angles between [-pi, pi)
            thetas = np.mod(thetas, 2.0*np.pi)
            thetas = np.where(thetas <= np.pi, thetas, thetas - 2.0*np.pi)
            # Sample from this            
            theta_coeffs = sample_from_a(thetas)
        elif params.drift_type == DRIFT_NONE:
            theta_coeffs = thetas
    
    alist.append([theta_coeffs, u_domain, params.H.real_term_flags[term] and params.reduce_dimension_flag])
    return S,b

def qnute_step(params: QNUTE_params, psi0):
    alist = []
    S_list = []
    b_list = []
    for i in range(params.H.num_terms - 1):
        sigma_expectation = tomography(params, psi0, alist, i)
        S,b = update_alist(params, sigma_expectation, alist, i, 0.5)
        S_list.append(S)
        b_list.append(b)
    
    sigma_expectation = tomography(params, psi0, alist, params.H.num_terms-1)
    S,b = update_alist(params, sigma_expectation, alist, params.H.num_terms-1, 1.0)
    S_list.append(S)
    b_list.append(b)

    for i in range(params.H.num_terms - 2, -1, -1):
        sigma_expectation = tomography(params, psi0, alist, i)
        S,b = update_alist(params, sigma_expectation, alist, i, 0.5)
        S_list.append(S)
        b_list.append(b)
    
    return propagate(params, psi0, alist), alist, S_list, b_list

def qnute(params: QNUTE_params, logging: bool=True):
    times = np.zeros(params.N + 1)
    svs = np.zeros((params.N+1, 2**params.nbits), dtype=complex)
    svs[0,:] = params.init_sv.data

    alist = []
    S_list = []
    b_list = []

    if logging: print('Starting Statevector QNUTE Simulation:')
    for i in range(1, params.N + 1):
        if logging: print('Iteration {}...'.format(i),end=' ',flush=True)
        
        t0 = time.time()

        psi, next_alist, next_slist, next_blist = qnute_step(params, Statevector(svs[i-1]))
        alist += next_alist
        S_list += next_slist
        b_list += next_blist
        svs[i,:] = psi.data

        t1 = time.time()
        duration = t1 - t0
        times[i] = duration
        
        if logging: print('Done -- Iteration time = {:0.2f} {}'.format(duration if duration < 60 else duration / 60, 'seconds' if duration < 60 else 'minutes'))
    
    return times, svs, alist, S_list, b_list

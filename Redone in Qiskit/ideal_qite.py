import numpy as np
import time

from helper import *
from qite_helpers import qite_params
import hamiltonians

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector

sv_sim = Aer.get_backend('statevector_simulator')

def evolve_statevector(qc, psi):
    '''
    Evolves the statevector psi through the circuit qc and returns the statevector
    '''
    nbits = int(np.log2(len(psi.data)))
    circ = QuantumCircuit(nbits)
    circ.initialize(psi)
    circ = circ + qc
    result = execute(circ, sv_sim).result()
    return result.get_statevector(circ)

def pauli_expectation(psi, p, qbits):
    '''
    returns the theoretical expectation <psi|P|psi> where P is the pauli string acting on qbits, indexed by p
    '''
    nbits = int(np.log2(len(psi.data)))
    pstring = int_to_base(p,4,len(qbits))
    qc = QuantumCircuit(nbits)
    for i in range(len(qbits)):
        if pstring[i] == 1:
            qc.x(qbits[i])
        elif pstring[i] == 2:
            qc.y(qbits[i])
        elif pstring[i] == 3:
            qc.z(qbits[i])
    
    # phi = psi.evolve(qc)
    phi = evolve_statevector(qc, psi)
    
    return np.real(np.vdot(psi.data, phi.data))

def measure_energy(psi, hm_list):
    E = 0.0
    for hm in hm_list:
        for j in range(len(hm[0])):
            E += pauli_expectation(psi, hm[0][j], hm[2]) * hm[1][j]
    return E

def propagate(qc, alist, params):
    for t in range(len(alist)):
        active = alist[t][1]
        nactive = len(active)
        if not params.odd_y_flag:
            for pstring in range(1,4**nactive):
                angle = np.real(alist[t][0][pstring])
                if np.abs(angle) > 1e-5:
                    pauli_string_exp(qc, active, pstring, angle)
        else:
            # Determine which hamiltonian term is being expanded from t
            # This goes up and down since the expansion is a 2nd order trotterization
            y = (t+1) % (2*params.nterms - 1)
            if y == 0:
                None
            elif y <= params.nterms:
                y = y-1
            else:
                y = 2*params.nterms - y - 1
            
            for i in range(len(params.odd_y_strings[y])):
                angle = np.real(alist[t][0][i])
                if np.abs(angle) > 1e-5:
                    pauli_string_exp(qc, active, params.odd_y_strings[y][i], angle)

def tomography(params, alist, term):
    sigma_expectation = {}
    
    qc = QuantumCircuit(params.nbits)
    # qc = params.init_circ.copy()
    propagate(qc, alist, params)
    psi = evolve_statevector(qc, params.init_sv)

    for key in params.measurement_keys[term]:
        if not isinstance(key, tuple):
            sigma_expectation[key] = pauli_expectation(psi, key, params.domains[term])
        else:
            p_,c_ = pauli_string_prod(key[0], key[1], key[2])
            full_domain = get_full_domain(params.hm_list[term][2], params.nbits )
            sigma_expectation[(key[0], key[1])] = pauli_expectation(psi, p_, full_domain) * c_

    return sigma_expectation

def update_alist(params, sigma_expectation, alist, term, scale):
    hm = params.hm_list[term]
    num_terms = len(hm[0])
    ndomain = len(params.domains[term])
    original_domain = get_full_domain(hm[2], params.nbits)

    if not params.odd_y_flag:
        nops = 4**ndomain
        S = np.zeros((nops,nops), dtype=complex)
        b = np.zeros(nops, dtype=complex)
        c = 1.0

        # Populate S
        for i in range(nops):
            for j in range(nops):
                p_,c_ = pauli_string_prod(i,j,ndomain)
                S[i,j] = sigma_expectation[p_] * c_
        
        # If the domain is at least as big as the original domain of the hamiltonian term
        if ndomain >= len(original_domain):
            # Calculate Norm
            for i in range(num_terms):
                c -= scale*params.db * hm[1][i] * sigma_expectation[ ext_domain_pauli(hm[0][i], hm[2], params.domains[term]) ]
            c = np.sqrt(c)
            
            # Populate b
            for i in range(nops):
                for j in range(num_terms):
                    ext_j = ext_domain_pauli(j, hm[2], params.domains[term])
                    p_,c_ = pauli_string_prod(i, ext_j, ndomain)
                    b[i] -= hm[1][j] * c_ * sigma_expectation[p_] / c
            b = -2.0 * np.imag(b)
        # If the domain is smaller than the original domain of the hamiltonian term
        else:
            # Calculate Norm
            for i in range(num_terms):
                c -= scale*params.db * hm[1][i] * sigma_expectation[ (0, ext_domain_pauli(hm[0][i], hm[2], original_domain) ) ]
            c = np.sqrt(c)
            
            # Populate b
            for i in range(nops):
                ext_i = ext_domain_pauli(i, params.domains[term], original_domain)
                for j in range(num_terms):
                    ext_j = ext_domain_pauli(j, hm[2], original_domain)
                    # Note that the coefficient for the tuple keys are already stored 
                    # in sigma_expectation
                    b[i] -= hm[1][j] * sigma_expectation[(ext_i, ext_j)] / c
            b = -2.0 * np.imag(b)
    else:
        nops = len(params.odd_y_strings[term])
        S = np.zeros((nops,nops), dtype=complex)
        b = np.zeros(nops, dtype=complex)
        c = 1.0

        # Populate S
        for i in range(nops):
            for j in range(nops):
                p_,c_ = pauli_string_prod(params.odd_y_strings[term][i],params.odd_y_strings[term][j],ndomain)
                S[i,j] = sigma_expectation[p_] * c_
        
        # If the domain is at least as large as the original domain
        if ndomain >= len(original_domain):
            # Calculate Norm
            for i in range(num_terms):
                c -= scale*params.db * hm[1][i] * sigma_expectation[ ext_domain_pauli(i, hm[2], params.domains[term]) ]
            c = np.sqrt(c)

            # Populate b
            for i in range(nops):
                ext_i = ext_domain_pauli(params.odd_y_strings[term][i], hm[2], params.domains[term])
                for j in range(num_terms):
                    ext_j = ext_domain_pauli(hm[0][j], hm[2], params.domains[term])
                    p_,c_ = pauli_string_prod(ext_i, ext_j, ndomain)
                    b[i] -= hm[1][j] * c_ * sigma_expectation[p_] / c
            b = -2.0 * np.imag(b)
        # If the domain is smaller than the original domain of the hamiltonian term
        else:
            # Calculate Norm
            for i in range(num_terms):
                c -= scale*params.db * hm[1][i] * sigma_expectation[ (0, ext_domain_pauli(hm[0][i], hm[2], original_domain)) ]
            c = np.sqrt(c)

            # Populate b
            for i in range(nops):
                ext_i = ext_domain_pauli( params.odd_y_strings[term][i], params.domains[term], original_domain )
                for j in range(num_terms):
                    ext_j = ext_domain_pauli(hm[0][j], hm[2], original_domain)
                    b[i] -= hm[1][j] * sigma_expectation[(ext_i, ext_j)] / c
            b = -2.0 * np.imag(b)

    
    # Add regularizer to make sure the system can be solved
    dalpha = np.eye(nops) * params.delta

    # Solve the system
    x = np.linalg.lstsq(2*np.real(S) + dalpha, -b, rcond=-1)[0]

    # Multiply by -2*db so that the appropriate rotation is applied since the sigma
    # rotation gates are exp(-theta/2 * sigma), and we want exp(a[I]*db * sigma_I)
    a_coefficients = -2.0 * params.db * x
    alist.append( [ a_coefficients, params.domains[term] ] )
    return c

def qite_step(params, alist):
    for i in range(params.nterms - 1):
        # simulate the step i
        sigma_expectation = tomography(params, alist, i)
        update_alist(params, sigma_expectation, alist, i, 1.0)
    
    # simulate the step (nterms-1)
    sigma_expectation = tomography(params, alist, params.nterms-1)
    update_alist(params, sigma_expectation, alist, params.nterms-1, 2.0)

    for i in range(params.nterms-2, -1, -1):
        #simulate backwards
        sigma_expectation = tomography(params, alist, i)
        update_alist(params, sigma_expectation, alist, i, 1.0)

    return alist

def qite(db, delta, N, nbits, D, hm_list, init_sv, time_flag):
    params = qite_params()
    params.initialize(hm_list, nbits, D)
    params.set_run_params(db, delta, 0, sv_sim, init_sv, None)

    E = np.zeros(N+1, dtype=complex)
    times = np.zeros(N+1,dtype=float)
    statevectors = np.zeros((N+1,2**nbits), dtype=complex)

    alist = []

    if init_sv is None:
        init_sv = Statevector.from_label('0'*nbits)
    else:
        init_sv = Statevector(init_sv)

    E[0] = measure_energy(init_sv, params.hm_list)
    statevectors[0] = init_sv.data

    print('Starting Ideal QITE Simulation:')
    for i in range(1,N+1):
        print('Iteration {}:...'.format(i), end='',flush=True)
        if time_flag:
            start = time.time()
        
        alist = qite_step(params, alist)

        qc = QuantumCircuit(nbits)
        propagate(qc, alist, params)
        psi = evolve_statevector(qc, params.init_sv)

        statevectors[i] = psi.data
        E[i] = measure_energy(psi, params.hm_list)

        print('Done',end='',flush=True)
        if time_flag:
            end = time.time()
            times[i] = end - start
            print(' -- Execution Time = {:0.2f} {}'.format( times[i] if times[i] < 60 else times[i] / 60, 'seconds' if times[i] < 60 else 'minutes' ))
        else:
            print()
    
    return E,times,statevectors

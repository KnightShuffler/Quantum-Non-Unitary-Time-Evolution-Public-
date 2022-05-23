from hamiltonians import TOLERANCE
import numpy as np
# Uncomment this if cupy is installed
# import cupy as cp
import time

from helpers import *
from qite_params import QITE_params

from qiskit import QuantumCircuit, execute
from qiskit.quantum_info import Statevector

# Note: These methods implicitly assumes that the backend is the Statevector simulator

# The tolerance for performing rotations
TOLERANCE = 1e-5

def evolve_statevector(params, qc, psi):
    '''
    Evolves the statevector psi through the circuit qc and returns the statevector
    '''
    circ = QuantumCircuit(params.nbits)
    circ.initialize(psi)
    circ = circ + qc
    result = execute(circ, params.backend).result()
    return result.get_statevector(circ)

def pauli_expectation(params, psi, p, qbits):
    '''
    returns the theoretical expectation <psi|P|psi> where P is the pauli string acting on qbits, indexed by p
    '''
    pstring = int_to_base(p,4,len(qbits))
    qc = QuantumCircuit(params.nbits)
    for i in range(len(qbits)):
        if pstring[i] == 1:
            qc.x(qbits[i])
        elif pstring[i] == 2:
            qc.y(qbits[i])
        elif pstring[i] == 3:
            qc.z(qbits[i])
    
    phi = evolve_statevector(params, qc, psi)
    
    return np.real(np.vdot(psi.data, phi.data))

def measure_energy(params, psi, hm_list):
    '''
    returns the mean energy <psi|H|psi>
    '''
    E = 0.0
    for hm in hm_list:
        for j in range(len(hm[0])):
            E += pauli_expectation(params, psi, hm[0][j], hm[2]) * hm[1][j]
    return E

def propagate(params, psi0, alist):
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

            angle = np.real(alist[t][0][i])
            if np.abs(angle) > TOLERANCE:
                pauli_string_exp(qc, domain, ops[i], angle)
    
    return evolve_statevector(params, qc, psi0)

def tomography(params, psi0, alist, term):
    sigma_expectation = {}

    psi = propagate(params, psi0, alist)

    if params.small_domain_flags[term]:
        domain = get_full_domain(params.hm_list[2], params.nbits)
    else:
        domain = params.domains[term]

    for key in params.measurement_keys[term]:
        sigma_expectation[key] = pauli_expectation(params, psi, key, domain)

    return sigma_expectation

def update_alist(params, sigma_expectation, alist, term, scale):
    hm = params.hm_list[term]
    num_terms = len(hm[0])
    domain = params.domains[term]
    ndomain = len(domain)
    active = get_full_domain(hm[2], params.nbits)
    
    if params.small_domain_flags[term]:
        big_domain = active
    else:
        big_domain = domain
    
    # Load c
    c = 1.0
    for j in range(num_terms):
        c -= 2*scale * params.db * sigma_expectation[ ext_domain_pauli(hm[0][j], hm[2], big_domain) ]
    
    # Load S
    if params.real_term_flags[term]:
        ops = params.odd_y_strings[ndomain]
    else:
        ops = list(range(4**ndomain))
    nops = len(ops)
    
    S = np.zeros((nops,nops), dtype=complex)
    for i in range(nops):
        I = ext_domain_pauli(ops[i], domain, big_domain)
        for j in range(nops):
            J = ext_domain_pauli(ops[j], domain, big_domain)
            p_,c_ = pauli_string_prod(I, J, len(big_domain))
            
            S[i,j] = sigma_expectation[p_] * c_
    
    # Load b
    b = np.zeros(nops, dtype=complex)
    for i in range(nops):
        I = ext_domain_pauli(ops[i], domain, big_domain)
        for j in range(num_terms):
            J = ext_domain_pauli(hm[0][j], hm[2], big_domain)

            p_,c_ = pauli_string_prod(I, J, len(big_domain))

            b[i] += scale * hm[1][j] * sigma_expectation[p_] * c_
    b = (4.0 / np.sqrt(c)) * np.imag(b)

    #Regularizer
    dalpha = np.eye(nops) * params.delta
    
    if not params.gpu_calc_flag:
        x = np.linalg.lstsq(2*np.real(S) + dalpha, b, rcond=-1)[0]
    else:
        S = cp.asarray(2*np.real(S) + dalpha)
        b = cp.asarray(b)
        x = cp.linalg.lstsq(S,b,rcond=-1)[0].get()
    
    a_coeffs = 2.0*params.db * x
    alist.append([a_coeffs, domain, params.real_term_flags[term]])

def qite_step(params, psi0):
    alist = []
    for i in range(params.nterms - 1):
        sigma_expectation = tomography(params, psi0, alist, i)
        update_alist(params, sigma_expectation, alist, i, 0.5)
    
    sigma_expectation = tomography(params, psi0, alist, params.nterms-1)
    update_alist(params, sigma_expectation, alist, params.nterms-1, 1.0)

    for i in range(params.nterms - 2, -1, -1):
        sigma_expectation = tomography(params, psi0, alist, i)
        update_alist(params, sigma_expectation, alist, i, 0.5)
    
    return propagate(params, psi0, alist), alist

def qite(params):
    E = np.zeros(params.N + 1)
    times = np.zeros(params.N + 1)
    statevectors = np.zeros((params.N+1, 2**params.nbits), dtype=complex)

    alist = []

    E[0] = measure_energy(params, params.init_sv, params.hm_list)
    statevectors[0] = params.init_sv.data

    print('Starting Ideal QITE Simulation:')
    for i in range(1, params.N + 1):
        print('Iteration {}...'.format(i),end=' ',flush=True)
        start = time.time()

        psi, next_alist = qite_step(params, Statevector(statevectors[i-1]))
        for a in next_alist:
            alist.append(a)

        statevectors[i] = psi.data
        E[i] = measure_energy(params, psi, params.hm_list)

        end = time.time()
        duration = end - start
        times[i] = duration
        print('Done -- Iteration time = {:0.2f} {}'.format(duration if duration < 60 else duration / 60, 'seconds' if duration < 60 else 'minutes'))
    
    return E, times, statevectors, alist

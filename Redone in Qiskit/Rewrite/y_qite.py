import numpy as np
from qiskit import Aer
from qiskit import QuantumCircuit

import time

from helper import *

def y_propagate(qc, alist):
    for t in range(len(alist)):
        active = alist[t][1]
        nbits = len(active)
        odd_y_paulis = odd_y_pauli_strings(nbits)
        for i in range(len(odd_y_paulis)):
            angle = np.real(alist[t][0][i])
            if np.abs(angle) > 1e-5:
                pauli_string_exp(qc, active, odd_y_paulis[i], angle)

def y_tomography(hm, alist, qbits, nbits, backend, shots=1024):
    nactive = len(qbits)
    odd_ys = odd_y_pauli_strings(nactive)
    nops = len(odd_ys)
    
    keys = []
    
    for j in hm[0]:
        if not(j in keys):
            keys.append(j)
    
    for i in odd_ys:
        for j in odd_ys:
            p,c = pauli_string_prod(i,j,nactive)
            if not (p in keys):
                keys.append(p)
        for j in hm[0]:
            p,c = pauli_string_prod(i,j,nactive)
            if not (p in keys):
                keys.append(p)
    
#     print(keys)
    sigma_expectation = {}
    for key in keys:
#         sigma_expectation[key] = 0.0j
        qc = QuantumCircuit(nbits, nbits)
        y_propagate(qc, alist)
        sigma_expectation[key],counts = measure(qc, key, qbits, backend, shots)
    
    return sigma_expectation

def y_update_alist(sigma_expectation, alist, db, delta, hm):
    nactive = len(hm[2])
    nterms = len(hm[0])
    
    odd_ys = odd_y_pauli_strings(nactive)
    nops = len(odd_ys)
    
    S = np.zeros([nops,nops],dtype=complex)
    for i in range(nops):
        for j in range(nops):
            p,c_ = pauli_string_prod(odd_ys[i], odd_ys[j], nactive)
            S[i,j] = sigma_expectation[p] * c_
    
    b = np.zeros(nops, dtype=complex)
    c = 1
    
    for i in range(nterms):
        c -= 2 * db * hm[1][i] * sigma_expectation[hm[0][i]]
    c = np.sqrt(c)
    
    
    for k in range(len(odd_ys)):
        # This part not needed since it will always be real
#         b[k] += (1/c - 1) * sigma_expectation[odd_ys[k]]
        for j in range(nterms):
            p,c_ = pauli_string_prod(odd_ys[k], hm[0][j], nactive)
            b[k] -= hm[1][j] * c_ * sigma_expectation[p] / c
    b = -2*np.imag(b)
    
    dalpha = np.eye(nops) * delta
    
    x = np.linalg.lstsq(2*np.real(S) + dalpha, -b, rcond=-1)[0]
    
    a_coefficients = -2*db*x
    
    alist.append([a_coefficients, hm[2]])
    
    return c

def y_measure_energy(alist, nbits, hm_list, backend, shots=1024):
    '''
    Measure the expected energy of the state evolved by alist, given the hm_list of the Hamiltonian
    '''
    Energy = 0
    
    # For each hm term
    for hm in hm_list:
        # For each pauli term in hm
        for j in range(len(hm[0])):
            qc = QuantumCircuit(nbits, nbits)
            # Get the state evolved from alist
            y_propagate(qc, alist)
            
            # Measure the expected value of the pauli term in this state
            expectation, counts = measure(qc, hm[0][j], hm[2], backend, num_shots=shots)
            
            # Add this with the relevant weight to the energy total
            Energy += hm[1][j] * expectation
    return Energy

def y_qite_step(alist, db, delta, nbits, hm_list, backend, shots=1024):
    '''
    performs one imaginary time step of size db in the QITE
    '''
    for hm in hm_list:
        # peform tomography on the active qubits in hm
        sigma_expectation = y_tomography(hm, alist, hm[2], nbits, backend, shots=1024)
        # obtain the next alist term corresponding to hm
        norm = y_update_alist(sigma_expectation, alist, db, delta, hm)
    return alist

def y_log_qite_run(db, delta, N, nbits, hm_list, alist, file_name='y_hm-alist'):
    logging = []
    logging.append(True)
    logging.append(db)
    logging.append(delta)
    logging.append(N)
    logging.append(nbits)
    logging.append(hm_list)
    logging.append(alist)
    
    np.save(file_name+'.npy',np.asarray(logging,dtype=object))

def y_qite(db, delta, N, nbits, hm_list, backend, shots=1024, details=False, log=False, log_file = 'y_hm-alist'):
    '''
    Performs the QITE algorithm with: 
        db: a time step,
        delta: a regularization,
        N: number of total iterations (time steps)
        nbits: number of qubits in the whole system
        hm_list: list of local terms in the Hamiltonian
        backend: the backend that the quantum circuit will run on
        shots: the number of shots for each circuit run
        details: flag to show details while the algorithm runs
    '''
    # stores the mean energy for each time step
    E = np.zeros(N+1,dtype=float)
    # stores the iteration times
    times = np.zeros(N+1,dtype=float)
    
    alist = []
    
    E[0] = y_measure_energy(alist, nbits, hm_list, backend, shots)
    
    start = 0
    end = 0
    if details:
        print('Starting QITE Loop')
    
    for i in range(1,N+1):
        if details:
            print('i={}:'.format(i), end='  ', flush=True)
            start = time.time()

        # correction_matrix = estimate_assignment_probs()
        alist = y_qite_step(alist, db, delta, nbits, hm_list, backend, shots)
        # Record the energy of the system after the QITE step
        E[i] = y_measure_energy(alist, nbits, hm_list, backend, shots)

        if details:
            end = time.time()
            duration = end-start
            times[i] = duration
            print('Execution time: {:.2f} seconds'.format(duration))

    if log:
        y_log_qite_run(db, delta, N, nbits, hm_list, alist, log_file)
    return E,times
import numpy as np
from qiskit import Aer
from qiskit import QuantumCircuit

aer_sim = Aer.get_backend('aer_simulator')

import time

from helper import *

def propogate(qc, alist):
    '''
    Applies the unitaries described by each time step's A
    '''
    for t in range(len(alist)):
        active = alist[t][1]
        nbits = len(active)
        for pstring in range(1,4**nbits):
            angle = np.real(alist[t][0][pstring])
            if np.abs(angle) > 1e-5:
                pauli_string_exp(qc, active, pstring, angle)
    
def update_alist(sigma_expectation, alist, db, delta, hm):
    '''
    updates the a term for hm given the tomography of the state prepared so far from sigma_expectation,
    returns the norm of the imaginary time evolved state
    '''
    # number of qubits that the term acts on
    nbits = len(hm[2])
    # number of pauli terms present in hm
    nterms = len(hm[0])
    
    nops = 4**nbits
    
    # Step 1: Obtain S matrix:
    # S[I,J] = <psi| sigma_I sigma_J |psi>
    
    S = np.zeros([nops,nops],dtype=complex)
    for i in range(nops):
        for j in range(nops):
            p,c_ = pauli_string_prod(i,j,nbits)
            S[i,j] = sigma_expectation[p]*c_
    
    # Step 2: Obtain b vector
    b = np.zeros(nops,dtype=complex)
    c = 1
    
    for i in range(nterms):
        c -= 2* db* hm[1][i] * sigma_expectation[hm[0][i]]
    c = np.sqrt(c)
    
    for i in range(nops):
        b[i] += ( 1/c - 1 ) * sigma_expectation[i] / db
        for j in range(nterms):
            p,c_ = pauli_string_prod(i,hm[0][j],nbits)
            b[i] -= hm[1][j] * c_ * sigma_expectation[p] / c
    b = -2*np.imag(b)
    
    # Step 3: Add regularizer to make sure the system can be solved
    dalpha = np.eye(nops)*delta
    
    # Step 4: Solve the system
    x = np.linalg.lstsq(2*np.real(S) + dalpha, -b, rcond=-1)[0]
    
    # Multiply by -2 db because the rotation gate is of the form exp(-i theta/2 P), but we want exp(i theta P)
    a_coefficients = -2*db*x
    # Append to the a coefficients
    alist.append([a_coefficients, hm[2]])
    
    return c

def measure_energy(alist, nbits, hm_list, backend, shots=1024):
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
            propogate(qc, alist)
            
            # Measure the expected value of the pauli term in this state
            expectation, counts = measure(qc, hm[0][j], hm[2], backend, num_shots=shots)
            
            # Add this with the relevant weight to the energy total
            Energy += hm[1][j] * expectation
    return Energy

def tomography(alist, qbits, nbits, backend, shots=1024):
    '''
    Perform tomography on the state evolved by alist
    '''
    nactive = len(qbits)
    nops = 4**nactive
    
    sigma_expectation = np.zeros(nops,dtype=float)
    
    # <psi|I|psi> = 1 for all states
    sigma_expectation[0] = 1
    
    for i in range(1,nops):
        qc = QuantumCircuit(nbits,nbits)
        # Get the state evolved from alist
        propogate(qc,alist)
        
        # measure the expectation of the pauli string indexed by i
        sigma_expectation[i],counts = measure(qc, i, qbits, backend, num_shots=shots)
    return sigma_expectation

def qite_step(alist, db, delta, nbits, hm_list, backend, shots=1024):
    '''
    performs one imaginary time step of size db in the QITE
    '''
    for hm in hm_list:
        # peform tomography on the active qubits in hm
        sigma_expectation = tomography(alist, hm[2], nbits, backend, shots)
        # obtain the next alist term corresponding to hm
        norm = update_alist(sigma_expectation, alist, db, delta, hm)
    return alist

def log_qite_run(db, delta, N, nbits, hm_list, alist, file_name='hm-alist'):
    logging = []
    logging.append(db)
    logging.append(delta)
    logging.append(N)
    logging.append(nbits)
    logging.append(hm_list)
    logging.append(alist)
    
    np.save(file_name+'.npy',np.asarray(logging,dtype=object)) 

def qite(db, delta, N, nbits, hm_list, backend, shots=1024, details=False, log=False, log_file = 'hm-alist'):
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
    
    E[0] = measure_energy(alist, nbits, hm_list, backend, shots)
    
    start = 0
    end = 0
    if details:
        print('Starting QITE Loop')
    
    for i in range(1,N+1):
        if details:
            print('i={}:'.format(i), end='  ', flush=True)
            start = time.time()

        # correction_matrix = estimate_assignment_probs()
        alist = qite_step(alist, db, delta, nbits, hm_list, backend, shots)
        # Record the energy of the system after the QITE step
        E[i] = measure_energy(alist, nbits, hm_list, backend, shots)

        if details:
            end = time.time()
            duration = end-start
            times[i] = duration
            print('Execution time: {:.2f} seconds'.format(duration))

    if log:
        log_qite_run(db, delta, N, nbits, hm_list, alist, log_file)
    return E,times

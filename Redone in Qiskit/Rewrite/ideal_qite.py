import numpy as np

from helper import *
from qite import propagate, update_alist, log_qite_run
from y_qite import y_propagate, y_update_alist, y_log_qite_run

from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Statevector

sv_sim = Aer.get_backend('statevector_simulator')

import matplotlib.pyplot as plt

from os import path, makedirs
import time

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
    
    phi = psi.evolve(qc)
    
    return np.real(np.vdot(psi.data, phi.data))

def ideal_energy(alist, nbits, hm_list, init_sv, odd_y_flag):
    Energy = 0
    qc = QuantumCircuit(nbits)
    if odd_y_flag:
        y_propagate(qc, alist)
    else:
        propagate(qc, alist)
    
    psi = Statevector(init_sv).evolve(qc)
    for hm in hm_list:
        for j in range(len(hm[0])):
            Energy += hm[1][j] * pauli_expectation(psi, hm[0][j], hm[2])
    return Energy

def ideal_tomography(hm, alist, qbits, nbits, init_sv, odd_y_flag):
    nactive = len(qbits)
    if odd_y_flag:
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
        
        sigma_expectation = {}
        for key in keys:
            qc = QuantumCircuit(nbits, nbits)
            y_propagate(qc, alist)
            psi = init_sv.evolve(qc)
            sigma_expectation[key] = pauli_expectation(psi, key, qbits)
        
        return sigma_expectation, psi
    else:
        nops=  4**nactive
        
        sigma_expectation = np.zeros(nops,dtype=float)
        sigma_expectation[0] = 1.0
        
        qc = QuantumCircuit(nbits)
        propagate(qc,alist)

        psi = init_sv.evolve(qc)
        
        for i in range(1,nops):
            sigma_expectation[i] = pauli_expectation(psi, i, qbits)
        return sigma_expectation, psi

def ideal_qite_step(alist, db, delta, nbits, hm_list, init_sv, odd_y_flag):
    for hm in hm_list:
        # get the ideal state tomography
        sigma_expectation, psi = ideal_tomography(hm, alist, hm[2], nbits, init_sv, odd_y_flag)
        if odd_y_flag:
            norm = y_update_alist(sigma_expectation, alist, db, delta, hm)
        else:
            norm = update_alist(sigma_expectation, alist, db, delta, hm)
    return alist, psi

def ideal_qite(db,delta,N,nbits,hm_list,init_sv,details=False,log=False,log_file='ideal-qite'):
    odd_y_flag = is_real_hamiltonian(hm_list)
    E = np.zeros(N+1,dtype=complex)
    times = np.zeros(N+1,dtype=float)
    
    states = []
    if init_sv == None:
        states.append(Statevector.from_label('0'*nbits))
    else:
        states.append(init_sv)
    
    alist = []
    
    E[0] = ideal_energy(alist, nbits, hm_list, states[0], odd_y_flag)
    
    start = 0
    end = 0
    if details:
        print('Starting QITE Loop')
    
    for i in range(1,N+1):
        if details:
            print('i={}'.format(i), end=' ', flush=True)
            start = time.time()
        
        alist,psi = ideal_qite_step(alist, db, delta, nbits, hm_list, states[0], odd_y_flag)
        E[i] = ideal_energy(alist, nbits, hm_list, states[0], odd_y_flag)
        states.append(psi)
        
        if details:
            end = time.time()
            duration = end-start
            times[i] = duration
            print('Execution time: {:.2f} seconds'.format(duration))
    
    if log:
        if odd_y_flag:
            y_log_qite_run(db,delta,N,nbits,hm_list,alist,log_file)
        else:
            log_qite_run(db,delta,N,nbits,hm_list,alist,log_file)
    
    return E,times,states

import numpy as np
from qiskit.quantum_info import Statevector

import time
from y_qite import y_log_qite_run
from qite import log_qite_run

from helper import *
from ideal_qite import ideal_energy, ideal_tomography

def update_alist_2nd_ord(sigma_expectation, alist, db, delta, hm, scale, odd_y_flag):
    # number of qubits that the term acts on
    nactive = len(hm[2])
    # number of pauli terms present in hm
    nterms = len(hm[0])
    
    if odd_y_flag:
        odd_ys = odd_y_pauli_strings(nactive)
        nops = len(odd_ys)

        # Step 1: Obtain S matrix:
        # S[I,J] = <psi| sigma_I sigma_J |psi>
        S = np.zeros([nops,nops],dtype=complex)
        for i in range(nops):
            for j in range(nops):
                p,c_ = pauli_string_prod(odd_ys[i], odd_ys[j], nactive)
                S[i,j] = sigma_expectation[p] * c_

        # Step 2: Obtain b vector
        b = np.zeros(nops, dtype=complex)
        c = 1

        for i in range(nterms):
            c -= scale*db*sigma_expectation[hm[0][i]]
        c = np.sqrt(c)

        for k in range(nops):
            # This part not needed since it will always be real
            # b[k] += (1/c - 1) * sigma_expectation[odd_ys[k]] / db
            for j in range(nterms):
                p,c_ = pauli_string_prod(odd_ys[k], hm[0][j], nactive)
                b[k] -= hm[1][j] * c_ * sigma_expectation[p] / c
        b = -2*np.imag(b)
    else:
        nops = 4**nactive

        # Step 1: Obtain S matrix:
        # S[I,J] = <psi| sigma_I sigma_J |psi>

        S = np.zeros([nops,nops],dtype=complex)
        for i in range(nops):
            for j in range(nops):
                p,c_ = pauli_string_prod(i,j,nactive)
                S[i,j] = sigma_expectation[p]*c_

        # Step 2: Obtain b vector
        b = np.zeros(nops,dtype=complex)
        c = 1

        for i in range(nterms):
            c -= scale* db* hm[1][i] * sigma_expectation[hm[0][i]]
        c = np.sqrt(c)

        for i in range(nops):
            # This part not needed since it will always be real
            # b[i] += ( 1/c - 1 ) * sigma_expectation[i] / db
            for j in range(nterms):
                p,c_ = pauli_string_prod(i,hm[0][j],nactive)
                b[i] -= hm[1][j] * c_ * sigma_expectation[p] / c
        b = -2*np.imag(b)

    # Step 3: Add regularizer to make sure the system can be solved
    dalpha = np.eye(nops)*delta

    # Step 4: Solve the system
    x = np.linalg.lstsq(2*np.real(S) + dalpha, -b, rcond=-1)[0]

    # Multiply by -2*db so that the appropriate rotation is applied since the rotation about sigma gates are exp(-theta/2 sigma) but we want exp(theta sigma)
    a_coefficients = -2.0*db*x
    # Append to the a coefficients
    alist.append([a_coefficients, hm[2]])

    return c

def ideal_qite_step_2nd_ord(alist, db, delta, nbits, hm_list, init_sv, odd_y_flag):
    nterms = len(hm_list)
    for m in range(nterms-1):
        hm = hm_list[m]
        sigma_expectation, psi = ideal_tomography(hm, alist, hm[2], nbits, init_sv, odd_y_flag)
        update_alist_2nd_ord(sigma_expectation, alist, db, delta, hm, 1.0, odd_y_flag)
    
    hm = hm_list[-1]
    sigma_expectation,psi = ideal_tomography(hm, alist, hm[2], nbits, init_sv, odd_y_flag)
    update_alist_2nd_ord(sigma_expectation, alist, db, delta, hm, 2.0, odd_y_flag)
    
    for m in range(nterms-1,-1,-1):
        hm = hm_list[m]
        sigma_expectation, psi = ideal_tomography(hm, alist, hm[2], nbits, init_sv, odd_y_flag)
        update_alist_2nd_ord(sigma_expectation, alist, db, delta, hm, 1.0, odd_y_flag)
    
    return alist, psi

def ideal_qite_2nd_ord(db,delta,N,nbits,hm_list,init_sv,details=False,log=False,log_file='ideal-qite'):
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
        
        alist,psi = ideal_qite_step_2nd_ord(alist, db, delta, nbits, hm_list, states[0], odd_y_flag)
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

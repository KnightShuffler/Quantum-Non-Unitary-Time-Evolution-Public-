import numpy as np
import time

from hamiltonians import *
from helpers import *
from qnute_params import QNUTE_params

def exp_mat_psi(mat, psi, truncate:int=-1):
    '''
    Calculates exp(mat)|psi> using the Taylor series of exp(mat)
    if truncate == -1, it will calculate the series up until the norm 
    the previous term above the accepted tolerance,
    else, if truncate == k > 0, it will calculate up to the k-th term of
    the Taylor series (mat)^k / k!
    '''
    chi = psi.copy()
    phi = psi.copy()
    i = 1
    while (truncate < 0 and np.linalg.norm(chi) > TOLERANCE) or (truncate >= 0 and i <= truncate) :
        chi = 1/i * (mat @ chi)
        phi += chi
        i += 1
    return phi

def update_alist(params: QNUTE_params, alist, term, psi0, truncate:int=-1):  
    H = params.H
    hm = H.hm_list[term]
    h_mat = H.get_term_submatrix(term)
    nbits = H.nbits
    nactive = len(hm[2])
    active = [0] * nactive
    for t in range(nactive):
        active[t] = H.map[ hm[2][t] ]
    nops = 4**nactive
    
    S = np.eye(nops, dtype=complex)
    b = np.zeros(nops, dtype=complex)
    for i in range(nops):
        for j in range(i):
            p_, c_ = pauli_string_prod(i, j, nactive)
            partial_pstring = int_to_base(p_, 4, nactive)
            full_pstring = [0] * nbits
            for k in range(nactive):
                full_pstring[active[k]] = partial_pstring[k]
            p_mat = sigma_matrices[full_pstring[0]]
            for k in range(1,nbits):
                p_mat = np.kron(sigma_matrices[full_pstring[k]], p_mat)
            
            S[i,j] = np.vdot(psi0, p_mat@psi0) * c_
            # S is Hermitian, so we know the upper triangle
            S[j,i] = S[i,j].conjugate()
        # The diagonal is full of 1s: <psi|I|psi>
        S[i,i] = 1.0
    
    psi_prime = exp_mat_psi(h_mat*params.dt, psi0, truncate=truncate)
    c = np.linalg.norm(psi_prime)
    for i in range(nops):
        partial_pstring = int_to_base(i, 4, nactive)
        full_pstring = [0] * nbits
        for k in range(nactive):
            full_pstring[active[k]] = partial_pstring[k]
        p_mat = sigma_matrices[full_pstring[0]]
        for k in range(1,nbits):
            p_mat = np.kron(sigma_matrices[full_pstring[k]], p_mat)
        b[i] = 2/(params.dt*c) * np.imag( np.vdot(psi0, p_mat @ psi_prime) )
    
    a = np.linalg.lstsq( 2*np.real(S), -b, rcond=-1 )[0]
    
    alist.append(a)

    return S, b

def qnute_step(params:QNUTE_params, psi0, truncate:int=-1, trotter_update:bool=False):
    alist = []
    slist = []
    blist = []
    psi = psi0.copy()
    H = params.H
    nbits = H.nbits
    for term in range(H.num_terms):
        S,b = update_alist(params, alist, term, psi, truncate)
        slist.append(S)
        blist.append(b)

        a = alist[-1]
        
        #update the state
        hm = H.hm_list[term]
        nactive = len(hm[2])
        active = [0] * nactive
        for t in range(nactive):
            active[t] = H.map[hm[2][t]]
        nops = 4**nactive
        A = np.zeros((2**nbits, 2**nbits), dtype=complex)
        for i in range(nops):
            partial_pstring = int_to_base(i, 4, nactive)
            full_pstring = [0] * nbits
            for k in range(nactive):
                full_pstring[active[k]] = partial_pstring[k]
            p_mat = sigma_matrices[full_pstring[0]]
            for k in range(1,nbits):
                p_mat = np.kron(sigma_matrices[full_pstring[k]], p_mat)
            A += a[i] * p_mat
            if trotter_update:
                psi = exp_mat_psi(-1j*a[i]*params.dt*p_mat, psi)
        if not trotter_update:
            psi = exp_mat_psi(-1j*A*params.dt, psi)            
    
    return alist, slist, blist, psi

def qnute(params:QNUTE_params, logging:bool=True, truncate:int=-1, trotter_update: bool=False):
    times = np.zeros(params.N+1)
    svs = np.zeros((params.N+1,2**params.nbits), dtype=complex)
    svs[0,:] = params.init_sv.data

    alist = []
    S_list = []
    b_list = []

    H = params.H
    
    for i in range(1,params.N+1):
        if logging: print('Iteration {}...'.format(i),end=' ', flush=True)
        
        t0 = time.time()
        
        next_alist, next_slist, next_blist, phi = qnute_step(params, svs[i-1], truncate, trotter_update)
        alist += next_alist
        S_list += next_slist
        b_list += next_blist
        svs[i,:] = phi
        
        t1 = time.time()
        duration = t1-t0
        times[i] = duration
        
        if logging: print('Done -- Iteration time = {:0.2f} {}'.format(duration if duration < 60 else duration/60, 'seconds' if duration < 60 else 'minutes'))
    return times, svs, alist, S_list, b_list
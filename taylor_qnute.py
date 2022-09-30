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

def update_alist(params: QNUTE_params, term, psi0, truncate:int=-1):  
    H = params.H
    hm = H.hm_list[term]
    h_mat = H.get_term_submatrix(term)
    nbits = H.nbits
    active = hm[2]
    nactive = len(hm[2])
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
            S[j,i] = S[i,j].conjugate()
    
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
        b[i] = -2/(params.dt*c) * np.imag( np.vdot(psi0, p_mat @ psi_prime) )
    
    a = np.linalg.lstsq( np.real(S), b, rcond=-1 )[0]
    
    return a, S, b

def qnute_step(params:QNUTE_params, psi0, truncate:int=-1, trotter_update:bool=False):
    alist = []
    slist = []
    blist = []
    psi = psi0.copy()
    H = params.H
    nbits = H.nbits
    for term in range(H.num_terms):
        a,S,b = update_alist(H, term, psi, truncate)
        alist.append(a)
        slist.append(S)
        blist.append(b)
        
        #update the state
        hm = H.hm_list[term]
        active = hm[2]
        nactive = len(active)
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

def qnute(params:QNUTE_params, psi0, N, logging:bool=True, truncate:int=-1, trotter_update: bool=False):
    alist = []
    slist = []
    blist = []
    svs = np.zeros((N+1,psi0.shape[0]), dtype=complex)
    svs[0,:] = psi0

    H = params.H
    
    for i in range(1,N+1):
        if logging: print('Iteration {}...'.format(i),end=' ', flush=True)
        t0 = time.time()
        n_a, n_s, n_b, phi = qnute_step(H, svs[i-1], truncate, trotter_update)
        alist += n_a
        slist += n_s
        blist += n_b
        svs[i,:] = phi
        t1 = time.time()
        duration = t1-t0
        if logging: print('Done -- Iteration time = {:0.2f} {}'.format(duration if duration < 60 else duration/60, 'seconds' if duration < 60 else 'minutes'))
    return alist, slist, blist, svs
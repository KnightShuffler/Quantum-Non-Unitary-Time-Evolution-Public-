import numpy as np

def short_range_heisenberg(nbits,J,B=0):
    hm_list = []
    for i in range(nbits-1):
        hm = [ [], [], [i,i+1] ]
        for j in range(3):
            hm[0].append( (j+1) + 4*(j+1) )
            hm[1].append(J[j])
        hm_list.append(hm)
    if B!=0:
        for i in range(nbits):
            hm_list.append([ [3], [B], [i] ])
    return hm_list

def long_range_heisenberg(nbits, J):
    hm_list = []
    for i in range(nbits):
        for j in range(nbits):
            if i==j:
                continue
            prefactor = 1/(np.abs(i-j)+1)
            hm = [ [],[],[i,j] ]
            for k in range(3):
                hm[0].append( (k+1) + 4*(k+1) )
                hm[1].append(prefactor * J[k])
            hm_list.append(hm)
    return hm_list

def afm_transverse_field_ising(nbits, J, h):
    hm_list = []
    for i in range(nbits-1):
        hm_list.append( [ [3 + 4*3], [J], [i,i+1] ] )
    for i in range(nbits):
        hm_list.append( [ [1], [h], [i] ] )
    return hm_list
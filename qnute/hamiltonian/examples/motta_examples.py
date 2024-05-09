import numpy as np
from .. import Hamiltonian

###################################
# Hamiltonian of Different Models #
###################################

class ShortRangeHeisenberg(Hamiltonian):
    def __init__(self, n_spins, J, B=0, n_dim=1):
        hm_list = []
        
        if n_dim != 1:
            raise ValueError('Short Range Heisenberg Model not implemented for more than 1D')

        if n_dim == 1:
            qubit_map = None
            bound = n_spins
            for i in range(n_spins-1):
                hm = [ [], [], [i,i+1] ]
                for j in range(3):
                    hm[0].append( (j+1) + 4*(j+1) )
                    hm[1].append(J[j])
                hm_list.append(hm)
            if B!=0:
                for i in range(n_spins):
                    hm_list.append([ [3], [B], [i] ])
        
        super().__init__(hm_list, n_dim, bound, qubit_map)

class LongRangeHeisenberg(Hamiltonian):
    def __init__(self,  n_spins, J, B=0, n_dim=1):
        hm_list = []

        if n_dim != 1:
            raise ValueError('Long Range Heisenberg Model not implemented for more than 1D')
        if n_dim == 1:
            qubit_map = None
            bound = n_spins
            for i in range(n_spins):
                for j in range(i+1, n_spins):
                    prefactor = 1/(np.abs(i-j)+1)
                    hm = [ [],[],[i,j] ]
                    for k in range(3):
                        hm[0].append( (k+1) + 4*(k+1) )
                        hm[1].append(prefactor * J[k])
                    hm_list.append(hm)
        
        super().__init__(hm_list, n_dim, bound, qubit_map)

class TransverseFieldIsing_AFM(Hamiltonian):
    def __init__(self, n_spins, J, h=0, n_dim=1):
        hm_list = []

        if n_dim != 1:
            raise ValueError('AFM Transverse Field Ising Model not implemented for more than 1D')
        
        if n_dim == 1:
            qubit_map = None
            bound = n_spins
            for i in range(n_spins-1):
                hm_list.append( [ [3 + 4*3], [J], [i,i+1] ] )
            if h != 0:
                for i in range(n_spins):
                    hm_list.append( [ [1], [h], [i] ] )

        super().__init__(hm_list, n_dim, bound, qubit_map)

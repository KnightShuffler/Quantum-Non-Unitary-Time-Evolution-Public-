import numpy as np

from helpers import *

# Format of the Hamiltonian:
# hm_list is a list of terms: [hm]
# hm is a term of the form: [ [pauli string ids], [amplitudes of pauli strings], [qubits that the pauli strings act on] ]
#   hm[2] is an ordered list of integers that are qubit indices, say [a_0,a_1,...,a_n]
#   hm[0] contains integers that when converted to a base-4 integer of n digits, the i-th digit tells us what pauli operator acts on qubit a_i
#   hm[1] is a list of complex numbers with the same length as hm[0] that has the amplitudes of each pauli string described in hm[0]
#   The term in the Hamiltonian is thus: Sum_i hm[1][i] * Pauli(hm[0][i]), which is an operator that acts on the qubits in hm[2]
# The Hamiltonian is a sum of all of the terms hm in hm_list
#
# Example: hm_list = [  [ [ 0, 3 ], [ 0.5, 0.75 ], [1] ],  
#                       [ [ 1*1 + 2*4 ], [ 0.33 ], [0,2] ] 
#                    ]
# represents the Hamiltonian:
# H = (0.5 I_1 + 0.75 Z_1) + ( 0.33 X_0 Y_2 )

####################
# Helper Functions #
####################

class Hamiltonian:
    def __init__(self, hm_list, lattice_dim, lattice_bound, qubit_map=None):
        self.hm_list = hm_list.copy()
        self.d = lattice_dim
        self.l = lattice_bound
        
        # None qubit_map corresponds to a default 1D mapping
        if qubit_map == None:
            if self.d != 1:
                raise ValueError('Default qubit map only available for 1D topology')
            self.map = {}
            for i in range(self.l):
                self.map[i] = i
        else:
            v = Hamiltonian.verify_map(lattice_dim, lattice_bound, qubit_map)
            if v != True:
                print('Qubit map not valid!')
                raise ValueError(v)
            self.map = qubit_map
        self.nbits = len(self.map)
        self.real_term_flags = self.is_real_hamiltonian()
    
    def verify_map(d, l, map):
        coords = map.keys()
        bound = l**d - 1
        counts = dict( (val, 0) for val in map.values())
        for coord in coords:
            if coord[0] > bound or coord[0] < 0 or coord[1] > bound or coord[1] < 0:
                return 'Out of bounds coordinate {}'.format(coord)
            counts[map[coord]] += 1
            if counts[map[coord]] > 1:
                return 'Multiple coordinates map to qubit index {}'.format(map[coord])
        return True

    def print(self):
        term = 0
        for hm in self.hm_list:
            term += 1
            print('Term {} acting on the qubit locations {}:'.format(term,hm[2]))
            for j in range(len(hm[0])):
                nactive = len(hm[2])
                pstring = int_to_base(hm[0][j],4,nactive)
                for i in range(nactive):
                    if pstring[i] == 0:
                        pstring[i] = 'I'
                    else:
                        pstring[i] = chr(ord('X') + pstring[i] - 1)
                print('\t({0:.2f} {1} {2:.2f}i)'.format(hm[1][j].real, '+-'[int(hm[1][j].imag) < 0], abs(hm[1][j].imag)),end=' ',flush=True)
                for i in range(nactive):
                    print('{}_{}'.format(pstring[i],hm[2][i]),end=' ',flush=True)
                if j < len(hm[0])-1:
                    print('+')
                else:
                    print()
    
    def get_matrix(self):
        '''
        returns the matrix representation of the Hamiltonian
        '''
        nbits = self.nbits
        num_basis = 2**nbits
        h_mat = np.zeros([num_basis,num_basis],dtype=complex)

        if nbits == 1:
            for hm in self.hm_list:
                for i in range(len(hm[0])):
                    h_mat += hm[1][i] * sigma_matrices[hm[0][i]]
        else:
            for hm in self.hm_list:
                active = [self.map[hm[2][i]] for i in range(len(hm[2]))]
                nactive = len(active)
                nterms = len(hm[0])
                # Loop through the Pauli terms in hm
                for i in range(nterms):
                    full_pauli_str = [0] * nbits
                    partial_pauli_str = int_to_base(hm[0][i],4,nactive)
                    for j in range(nactive):
                        full_pauli_str[active[j]] = partial_pauli_str[j]
                    # reverse the string to be consistend with Qiskit's qubit ordering
                    full_pauli_str = full_pauli_str[::-1]
                    # The matrix for the term is a tensor product of the corresponding Pauli matrices
                    term_matrix = sigma_matrices[full_pauli_str[0]]
                    for j in range(1,nbits):
                        term_matrix = np.kron(term_matrix, sigma_matrices[full_pauli_str[j]])
                    # Scale by the coefficient of the term
                    term_matrix *= hm[1][i]
                    
                    # Add the term to the final matrix
                    h_mat += term_matrix
        return h_mat

    def get_spectrum(self):
        '''
        returns the spectrum of the Hamiltonian
        '''
        h_mat = self.get_matrix()
        return np.linalg.eig(h_mat)

    def get_gs(self):
        '''
        returns the ground state energy and the ground state vector of the Hamiltonian
        '''
        w,v = self.get_spectrum()
        i = np.argmin(w)
        return w[i],v[:,i]
    
    def is_real_hamiltonian(self):
        '''
        returns whether each term of the hamiltonian is a real matrix in the Z basis
        '''
        real_flags = [True] * len(self.hm_list)
        for m in range(len(self.hm_list)):
            hm = self.hm_list[m]
            nactive = len(hm[2])
            odd_ys = odd_y_pauli_strings(nactive)
            for j in range(len(hm[0])):
                # If a term with odd Ys, the coefficient should be imaginary
                if hm[0][j] in odd_ys:
                    if np.abs(np.real(hm[1][j])) > TOLERANCE:
                        real_flags[m] = False
                        break
                # If a term with even Ys, the coefficient should be real
                else:
                    if np.abs(np.imag(hm[1][j])) > TOLERANCE:
                        real_flags[m] = False
                        break
        return real_flags
    
    ...

def get_k(hm_list):
    '''
    Given that hm_list describes a k-local Hamiltonian on a 
    linear qubit topology, this function returns what k is.
    '''
    k = 0
    for hm in hm_list:
        range = np.max(hm[2]) - np.min(hm[2]) + 1
        if k < range:
            k = range
    return k

def is_valid_domain(hm_list, D, nbits):
    '''
    Checks if D is a valid domain size for a k-local Hamiltonian
    on a linear qubit topology
    '''
    # domain size is restrited to the number of qubits
    if D > nbits:
        return False
    
    k = get_k(hm_list)
    # Only a valid domain size if D and k have the same parity
    return k%2 == D%2

def is_real_hamiltonian(hm_list):
    '''
    returns whether the described hamiltonian is a real matrix in the Z basis
    '''
    real_flags = [True] * len(hm_list)
    for m in range(len(hm_list)):
        hm = hm_list[m]
        nactive = len(hm[2])
        odd_ys = odd_y_pauli_strings(nactive)
        for j in range(len(hm[0])):
            # If a term with odd Ys, the coefficient should be imaginary
            if hm[0][j] in odd_ys:
                if np.abs(np.real(hm[1][j])) > TOLERANCE:
                    real_flags[m] = False
                    break
            # If a term with even Ys, the coefficient should be real
            else:
                if np.abs(np.imag(hm[1][j])) > TOLERANCE:
                    real_flags[m] = False
                    break
    return real_flags

###################################
# Hamiltonian of Different Models #
###################################

def short_range_heisenberg(nbits,J,B=0.0):
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
        for j in range(i+1, nbits):
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
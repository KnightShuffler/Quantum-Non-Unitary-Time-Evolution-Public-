from ast import literal_eval
import numpy as np
from qnute.helpers import int_to_base
from qnute.helpers import TOLERANCE
from qnute.helpers.lattice import in_lattice
from qnute.helpers.pauli import sigma_matrices
from qnute.helpers.pauli import odd_y_pauli_strings

# Format of the Hamiltonian:
#   hm_list is a list of terms: [hm]
#   hm is a term of the form: [ [pauli string ids], [amplitudes of pauli strings], [qubits that the pauli strings act on] ]
#       hm[2] is an ordered list of integers or integer tuples that are qubit coordinates, say [a_0,a_1,...,a_n]
#       hm[0] contains integers that when converted to a base-4 integer of n digits, the i-th digit tells us what pauli operator acts on qubit a_i
#       hm[1] is a list of complex numbers with the same length as hm[0] that has the amplitudes of each pauli string described in hm[0]
#       The term in the Hamiltonian is thus: Sum_i hm[1][i] * Pauli(hm[0][i]), which is an operator that acts on the qubits in hm[2]
#   The Hamiltonian is a sum of all of the terms hm in hm_list
#
#   Example: hm_list = [ [ [ 0, 3 ], [ 0.5, 0.75 ], [1] ],  
#                        [ [ 1*1 + 2*4 ], [ 0.33 ], [0,2] ] 
#                    ]
#   represents the Hamiltonian:
#   H = (0.5 I_1 + 0.75 Z_1) + ( 0.33 X_0 Y_2 )
# 
#   d = lattice_dim represents the number of dimensions that the qubits live in, 
#   this dictates the dimension of the tuples in hm[2], if lattice_dim == 1, then
#   hm[2] can be populated with integers
#
#   l = lattice_bound represents the side length of the bounding box of the qubit
#   lattice. Coordinates in hm[2] can only be in the range 0 <= i < l
#
#   qubit_map: A dictionary that maps qubit coordinates to the index of the logical
#   qubit in the circuit, for 1-D lattices, if the map is left None, it will map the
#   coordinate directly to the logical qubit index

hm_dtype = np.dtype([('pauli_id',np.uint32), ('amplitude', np.complex128)])

class Hamiltonian:
    def __init__(self, hm_list, lattice_dim, lattice_bound, qubit_map=None):
        # self.hm_list = hm_list.copy()
        self.hm_list = Hamiltonian.generate_ham_list(hm_list)
        self.num_terms = len(hm_list)
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
    
    @staticmethod
    def generate_ham_list(hm_list):
        numterms = 0
        for hm in hm_list:
            numterms += len(hm[0])
        ham_list = np.zeros(numterms,dtype=hm_dtype)
        i = 0
        for hm in hm_list:
            for j in range(len(hm[0])):
                ham_list[i] = (hm[0][j],hm[1][j])
                i += 1
        return ham_list

    def verify_map(d, l, map):
        coords = map.keys()
        counts = dict( (val, 0) for val in map.values())
        for coord in coords:
            if not in_lattice(coord, d, l):
                return 'Out of bounds coordinate {}'.format(coord)
            counts[map[coord]] += 1
            if counts[map[coord]] > 1:
                return 'Multiple coordinates map to qubit index {}'.format(map[coord])
        return True

    def print(self):
        '''
        Prints all the terms of the Hamiltonian
        '''
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
    
    def multiply_scalar(self, scalar):
        '''
        multiplies a scalar value to the Hamiltonian
        '''
        for i in range(len(self.hm_list)):
            for j in range(len(self.hm_list[i][1])):
                self.hm_list[i][1][j] *= scalar

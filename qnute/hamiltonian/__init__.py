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
        self.pterm_list, self.hm_indices = Hamiltonian.generate_ham_list(hm_list, self.map)
        self.num_terms = len(self.hm_indices)
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
    def generate_ham_list(hm_list, qubit_map):
        num_pterms = np.uint(0)
        num_terms = len(hm_list)
        hm_indices = np.zeros(num_terms, dtype=np.uint)
        for i,hm in enumerate(hm_list):
            hm_indices[i] = num_pterms
            num_pterms += len(hm[0])
        p_list = np.zeros(num_pterms,dtype=hm_dtype)
        
        i = 0
        for hm in hm_list:
            for j in range(len(hm[0])):
                p_list[i] = (hm[0][j], hm[1][j])
                i += 1
        return p_list, hm_indices

    @staticmethod
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
    
    def __str__(self):
        r_str = 'Hamiltonian Pauli Terms and Amplitudes:\n'
        for pterm in self.pterm_list:
            pstring = ''
            for i,p in enumerate(int_to_base(pterm['pauli_id'],4,self.nbits)):
                if p == 0:
                    pstring += 'I'
                else:
                    pstring += chr(ord('X')+p-1)
                pstring += f'_{i} '
            r_str += f'\t{pstring} : ({pterm["amplitude"]:.5f})\n'
        return r_str
    
    def get_matrix(self):
        '''
        returns the matrix representation of the Hamiltonian
        '''
        N = 2**self.nbits
        h_mat = np.zeros((N,N),dtype=np.complex128)
        
        for pterm in self.pterm_list:
            pdigits = int_to_base(pterm['pauli_id'], 4, self.nbits)
            term_mat = np.ones((1,1),dtype=np.complex128)
            for p in pdigits:
                term_mat = np.kron(sigma_matrices[p], term_mat)
            h_mat += term_mat * pterm['amplitude']
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
        self.pterm_list[:]['amplitude'] *= scalar

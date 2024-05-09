import numpy as np
from numba import njit
from itertools import combinations
from copy import deepcopy
from math import isclose
from numbers import Number
from qnute.helpers import int_to_base
from qnute.helpers.pauli import ext_domain_pauli
from qnute.helpers.pauli import get_pauli_prod_matrix
from qnute.helpers.pauli import odd_y_pauli_strings
from qnute.helpers.pauli import pauli_string_prod

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

def hm_list_adjoint(hm_list):
    return [[hm[0], np.conjugate(hm[1]), hm[2]] for hm in hm_list]

@njit
def hm_pterm_tensor(hm1_ind, hm1_amp, hm2_ind, hm2_amp, len_d2):
    hm_ind = np.zeros(hm1_ind.shape[0]*hm2_ind.shape[0], dtype=np.uint32)
    hm_amp = np.zeros(hm1_ind.shape[0]*hm2_ind.shape[0], dtype=np.complex128)
    k = 0
    for i,p1 in enumerate(hm1_ind):
        a1 = hm1_amp[i]
        for j,p2 in enumerate(hm2_ind):
            a2 = hm2_amp[j]
            hm_ind[k] = (p1*np.left_shift(1, len_d2*2) + p2)
            hm_amp[k] = (a1*a2)
            k += 1
    return hm_ind, hm_amp

def hm_list_tensor(hm_list1, hm_list2, num_qbits1:int=None, num_qbits2:int=None):
    if num_qbits1 is None:
        num_qbits1 = 0
        for hm in hm_list1:
            if (q:=np.max(hm[2])) > num_qbits1:
                num_qbits1 = q
        num_qbits1 += 1
    if num_qbits2 is None:
        num_qbits2 = 0
        for hm in hm_list2:
            if (q:=np.max(hm[2])) > num_qbits2:
                num_qbits2 = q
        num_qbits2 += 1
    
    assert num_qbits1 != 0
    assert num_qbits2 != 0

    hm_list = []
    for hm1 in hm_list1:
        for hm2 in hm_list2:
            len_d2 = len(hm2[2])
            hm = [None, 
                  None,
                  np.concatenate((hm2[2], hm1[2]+num_qbits2))]
            hm[0],hm[1] = hm_pterm_tensor(hm1[0], hm1[1], hm2[0], hm2[1], len_d2)
            hm_list.append(hm)
    return hm_list

@njit
def add_hm_terms(hm1_ind, hm1_amp, hm2_ind, hm2_amp):
    hm_ind = np.zeros(hm1_ind.shape[0] + hm2_ind.shape[0],dtype=np.uint32)
    hm_amp = np.zeros(hm1_ind.shape[0] + hm2_ind.shape[0],dtype=np.complex128)
    
    num_terms = hm1_ind.shape[0]
    hm_ind[0:num_terms] = hm1_ind
    hm_amp[0:num_terms] = hm1_amp
    

    for j,p2 in enumerate(hm2_ind):
        if p2 not in hm_ind[0:num_terms]:
            hm_ind[num_terms] = p2
            hm_amp[num_terms] = hm2_amp[j]
            num_terms += 1
        else:
            k = np.where(hm_ind[0:num_terms] == p2)[0]
            hm_amp[k] += hm2_amp[j]
    
    # hm_ind.reshape(num_terms)
    # hm_amp.reshape(num_terms)
    non_zeros = np.where(hm_amp != 0.0j)[0]
    return hm_ind[non_zeros], hm_amp[non_zeros]
            
def hm_list_add(hm_list1, hm_list2):
    assert not hm_list1 is None
    if hm_list2 is None:
        return hm_list1
    hm_list = []
    terms_added = [set(),set()] # For hm_list2
    for i,hm1 in enumerate(hm_list1):
        for j,hm2 in enumerate(hm_list2):
            if j not in terms_added[1]:
                if hm1[2].shape == hm2[2].shape:
                    if (hm1[2] == hm2[2]).all():
                        hm = [None,None,hm2[2]]
                        hm[0],hm[1] = add_hm_terms(hm1[0], hm1[1], hm2[0], hm2[1])
                        hm_list.append(hm)
                        terms_added[0].add(i)
                        terms_added[1].add(j)
    for i,hm1 in enumerate(hm_list1):
        if i not in terms_added[0]:
            hm_list.append(hm1)
    for j,hm2 in enumerate(hm_list2):
        if j not in terms_added[1]:
            hm_list.append(hm2)
    return hm_list

def hm_list_prod(hm_list1,hm_list2):
    assert not hm_list1 is None
    if hm_list2 is None:
        return hm_list1
    hm_list = []
    for hm1 in hm_list1:
        for hm2 in hm_list2:
            term_domain = list(set(hm1[2])|set(hm2[2]))
            num_terms = hm1[0].shape[0] * hm2[0].shape[0]
            hm = [np.zeros(num_terms,np.uint32), np.zeros(num_terms,np.complex128), term_domain]

            index = 0
            for i,pi in enumerate(hm1[0]):
                PI = ext_domain_pauli(pi, hm1[2], term_domain)
                for j,pj in enumerate(hm2[0]):
                    PJ = ext_domain_pauli(pj, hm2[2], term_domain)
                    P,C = pauli_string_prod(PI,PJ,len(term_domain))
                    hm[0][index] = P
                    hm[1][index] = hm1[1][i] * hm2[1][j] * C
                    index += 1

            hm_list.append(hm)
    return hm_list

def hm_list_sum(*args):
    if len(args) == 1:
        return args[0]
    if len(args) == 2:
        return hm_list_add(args[0], args[1])
    hm_list = args[0]
    for i in range(1,len(args)):
        hm_list = hm_list_add(hm_list, args[i])
    return hm_list

def get_identity_hm_list(num_qbits:int, scalar:Number|np.number=1.0):
    return [[np.zeros(1,dtype=np.uint32), np.ones(1,dtype=np.complex128)*scalar, np.arange(num_qbits)]]

class Hamiltonian:
    def __init__(self, hm_list, nbits):
        self.hm_list = Hamiltonian.reduce_hm_list(hm_list, nbits)
        self.pterm_list, self.hm_indices = Hamiltonian.generate_ham_list(self.hm_list, nbits)
        self.num_terms = len(self.hm_list)
        self.nbits = nbits
        # self.nbits = len(qubit_map)
        self.real_term_flags = None
    
    @staticmethod
    def reduce_hm_list(hm_list, nbits):
        new_list = []
        d = {}
        for r in range(1,nbits+1):
            for c in combinations(range(nbits),r):
                d[c] = [[],[]]
        identity_amplitude = 0.0j
        for hm in hm_list:
            term_domain = np.array(hm[2])
            for i,p in enumerate(hm[0]):
                if hm[1][i] == 0.0j:
                    continue
                if p == 0:
                    identity_amplitude += hm[1][i]
                    continue
                p_digits = np.array(int_to_base(p, 4, len(term_domain)))
                term_subdomain = tuple(term_domain[np.nonzero(p_digits)[0]])
                new_p = 0
                j = 0
                for dig in p_digits:
                    if dig != 0:
                        new_p += dig*(4**j)
                        j+=1
                try:
                    index = d[term_subdomain][0].index(new_p)
                    d[term_subdomain][1][index] += hm[1][i]
                except ValueError:
                    d[term_subdomain][0].append(new_p)
                    d[term_subdomain][1].append(hm[1][i])
        
        if identity_amplitude != 0.0j:
            d[tuple(range(nbits))][0].append(0)
            d[tuple(range(nbits))][1].append(identity_amplitude)
        
        for term_domain, id_amp in d.items():
            if len(id_amp[0]) == 0:
                continue
            new_list.append([
                np.array(id_amp[0], dtype=np.uint32),
                np.array(id_amp[1], dtype=np.complex128),
                np.array(term_domain, dtype=np.uint32)
            ])
        return new_list

    @staticmethod
    def generate_ham_list(hm_list, nbits):
        # nbits = len(qubit_map)
        num_pterms = np.uint32(0)
        num_terms = len(hm_list)
        hm_indices = np.zeros(num_terms, dtype=np.uint32)
        for i,hm in enumerate(hm_list):
            hm_indices[i] = num_pterms
            num_pterms += len(hm[0])
        p_list = np.zeros(num_pterms,dtype=hm_dtype)
        
        i = 0
        for hm in hm_list:
            active_qubits = hm[2]
            for j in range(len(hm[0])):
                p_list[i] = (ext_domain_pauli(hm[0][j], active_qubits, list(range(nbits))), hm[1][j])
                i += 1
        return p_list, hm_indices
    
    def __str__(self):
        r_str = 'Hamiltonian Pauli Terms and Amplitudes:\n'
        for pi,pterm in enumerate(self.pterm_list):
            pstring = ''
            for i,p in enumerate(int_to_base(pterm['pauli_id'],4,self.nbits)[::-1]):
                if p == 0:
                    pstring += 'I'
                else:
                    pstring += chr(ord('X')+p-1)
                pstring += f'_{self.nbits-i-1} '
            if pi in self.hm_indices:
                r_str += '\n'
            r_str += f'\t{pstring} : ({pterm["amplitude"]:.5f})\n'
        return r_str
    
    def get_matrix(self):
        '''
        returns the matrix representation of the Hamiltonian
        '''
        N = 2**self.nbits
        h_mat = np.zeros((N,N),dtype=np.complex128)
        
        for pterm in self.pterm_list:
            term_mat = get_pauli_prod_matrix(pterm['pauli_id'], self.nbits)
            h_mat += term_mat * pterm['amplitude']
        return h_mat

    def get_spectrum(self):
        '''
        returns the spectrum of the Hamiltonian
        '''
        h_mat = self.get_matrix()
        eig_vals, eig_states = np.linalg.eig(h_mat)
        idx = eig_vals.argsort()[::-1]   
        eig_vals = eig_vals[idx]
        eig_states = eig_states[:,idx]
        return eig_vals, eig_states

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
    
    def get_hm_pterms(self, term):
        assert term < self.num_terms and term >= 0
        return self.pterm_list[self.hm_indices[term]: self.hm_indices[term+1] if term+1 < self.num_terms else None]
    
    def __add__(self, other):
        if isinstance(other, Hamiltonian):
            hm_list = hm_list_add(self.hm_list, other.hm_list)
            nbits = np.max([self.nbits, other.nbits])
            return Hamiltonian(hm_list, nbits)
        elif isinstance(other, Number|np.number):
            hm_list = hm_list_add(self.hm_list, get_identity_hm_list(self.nbits, other))
            return Hamiltonian(hm_list, self.nbits)
        else:
            raise TypeError(f'Type {type(other)} is not supported in the Hamiltonian sum!')
    
    def __mul__(self, other):
        if isinstance(other, Number):
            H = deepcopy(self)
            H.multiply_scalar(other)
            return H
        elif isinstance(other, Hamiltonian):
            return Hamiltonian(hm_list_prod(self.hm_list, other.hm_list), self.nbits)
        else:
            raise TypeError(f'Type {type(other)} is not supported in the Hamiltonian product!')
        
    def __imul__(self, other):
        if isinstance(other, Number):
            self.multiply_scalar(other)
            return self
        elif isinstance(other, Hamiltonian):
            return Hamiltonian(hm_list_prod(self.hm_list, other.hm_list), self.nbits)
        else:
            raise TypeError(f'Type {type(other)} is not supported in the Hamiltonian product!')
    
    @staticmethod
    def adjoint(H:'Hamiltonian')->'Hamiltonian':
        return Hamiltonian(hm_list_adjoint(H.hm_list), H.nbits)
    
    @staticmethod
    def tensor_product(H1:'Hamiltonian', H2:'Hamiltonian')->'Hamiltonian':
        hm_list = hm_list_tensor(H1.hm_list, H2.hm_list, H1.nbits, H2.nbits)
        nbits = H1.nbits + H2.nbits
        return Hamiltonian(hm_list, nbits)
    
    @staticmethod
    def tensor_product_multi(*hams) -> 'Hamiltonian':
        for i,ham in enumerate(hams):
            assert isinstance(ham, Hamiltonian), 'Arguments must be Hamiltonian objects'
            if i == 0:
                H = ham
            else:
                H = Hamiltonian.tensor_product(ham, H)
        return H
    
    def rearrange_terms(self, u_domains:list[set[int]], amplitude_splits:np.ndarray[np.ndarray[float]]) -> 'Hamiltonian':
        new_pterm_list = np.zeros(len(u_domains)*self.pterm_list.shape[0], dtype=hm_dtype)
        new_hm_indices = []
        counter = 0
        for i,dom in enumerate(u_domains):
            new_hm_indices.append(counter)
            for term in range(self.num_terms):
                for pterm in self.get_hm_pterms(term):
                    new_amplitude = pterm['amplitude']*amplitude_splits[term,i]
                    if new_amplitude != 0.0:
                        new_pterm_list[counter]=(pterm['pauli_id'], pterm['amplitude']*amplitude_splits[term,i])
                        counter += 1

        # new_pterm_list.resize(counter)
        
        ham2 = deepcopy(self)
        ham2.pterm_list = new_pterm_list[0:counter]
        ham2.hm_indices = new_hm_indices
        ham2.num_terms = len(u_domains)

        return ham2
    
    def get_hm_term_support(self, term:int) -> set[int]:
        p = self.pterm_list[self.hm_indices[term]]['pauli_id']
        if p == 0:
            return set(range(self.nbits))
        support = set()
        for i in range(self.nbits):
            if p % 4 != 0:
                support.add(i)
            p //= 4
        return support

    def calculate_real_terms(self):
        self.real_term_flags = [True] * self.num_terms
        odd_y = odd_y_pauli_strings(self.nbits)
        term = 0
        for i,pterm in enumerate(self.pterm_list):
            if term < self.num_terms - 1:
                if i == self.hm_indices[term+1]:
                    term += 1
            if pterm['pauli_id'] in odd_y:
                self.real_term_flags[term] &= isclose(pterm['amplitude'].real, 0.0)
            else:
                self.real_term_flags[term] &= isclose(pterm['amplitude'].imag, 0.0)
    
    @staticmethod
    def Identity(num_qbits)->'Hamiltonian':
        return Hamiltonian(get_identity_hm_list(num_qbits), num_qbits)
            


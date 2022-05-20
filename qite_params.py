import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import hamiltonians
from helpers import *

class QITE_params:
    def __init__(self):
        # Hamiltonian Information
        self.hm_list = []
        self.nterms = 0
        self.real_term_flags = []
        self.odd_y_strings = {}
        self.domains = []
        self.measurement_keys = {}

        # QITE Run Parameters
        self.db = 0.0
        self.delta = 0.0
        self.N = 0
        self.num_shots = 0
        self.nbits = 0
        self.D = 0
        self.init_sv = None
        self.init_circ = None

        # GPU usage Flags
        self.gpu_simulator_flag = False
        self.gpu_calculation_flag = False

        # Logging Information
        self.log_path = ''
        self.fig_path = ''
        self.run_name = ''
    
    def domain_size(qbits):
        return max(qbits) - min(qbits) + 1

    def get_extended_domain(active, D, nbits):
        '''
        Returns a list of the extended domain of a k-local Hamiltonian term hm
        with a domain size D, and nbits qubits in a linear topology
        
        Note: D must be greater than the range of the hamiltonian term
        '''
        if QITE_params.domain_size(active) % 2 != D % 2:
            D -= 1
        radius = D//2
        center = (min(active) + max(active)) // 2
        _max = min(center + radius, nbits)
        _min = max(center + radius - D + 1, 0)

    def hamiltonian_precomputations(self, hm_list, nbits, D):
        # Validate the passed parameters
        if not hamiltonians.is_valid_domain(hm_list, D, nbits):
            raise ValueError('The domain size D is not valid for the hamiltonian, parameters not set.')

        self.hm_list = hm_list
        self.nterms = len(hm_list)
        self.nbits = nbits
        self.D = D

        # Calculate the domains of the unitaries simulating each term
        # If the domain size is less than the number of qubits calculate the domains
        if D != nbits:
            for hm in self.hm_list:
                self.domains.append(self.extended_domain(hm[2], self.D, self.nbits))
        # Otherwise, set the domain to be the full range of qubits
        else:
            self.domains = [list(range(0,nbits))] * self.nterms

        # Check if the terms are real
        self.real_term_flags = hamiltonians.is_real_hamiltonian(self.hm_list)

        # Initialize the keys for the odd y strings
        for m in range(self.nterms):
            if self.real_term_flags[m]:
                self.odd_y_strings[len(self.domains[m])] = None
        
        # Load the odd Y Pauli Strings
        for y_len in self.odd_y_strings.keys():
            self.odd_y_strings = odd_y_pauli_strings(y_len)
        
        # Calculate the strings to measure for each term:
        for m in range(self.nterms):
            # Initialize list of keys for the m-th term
            self.measurement_keys[m] = []
            ndomain = len(self.domains[m])
            # Populate the keys
            domain_ops = self.odd_y_strings[ndomain] if self.real_term_flags[m] else list(range(4**ndomain))
            self.get_measurement_keys(m, domain_ops)

    def get_measurement_keys(self, m, domain_ops):
        hm = self.hm_list[m]
        active = np.arange(self.domain_size(hm[2])) + min(hm[2])
        domain = self.domains[m]
        ndomain = len(domain)
        nactive = len(active)
        
        if ndomain >= nactive:
            big_domain = domain
        else:
            big_domain = active

        # Measurements for c: Pauli strings in hm
        for j in hm[0]:
            J = ext_domain_pauli(j, hm[2], big_domain)
            if J not in self.measurement_keys[m]:
                self.measurement_keys[m].append(J)
        
        # Measurements for S: Products of Pauli strings on the unitary domain
        for i in domain_ops:
            I = ext_domain_pauli(i, domain, big_domain)
            for j in domain_ops:
                J = ext_domain_pauli(j, domain, big_domain)
                
                p_,c_ = pauli_string_prod(I, J, len(big_domain))
                if p_ not in self.measurement_keys[m]:
                    self.measurement_keys[m].append(p_)
        
        # Measurements for b: Products of Pauli strings on the unitary domain with 
        # the Pauli strings in hm
        for i in domain_ops:
            I = ext_domain_pauli(i, domain, big_domain)
            for j in hm[0]:
                J = ext_domain_pauli(j, hm[2], big_domain)
                
                p_,c_ = pauli_string_prod(I, J, len(big_domain))
                if p_ not in self.measurement_keys[m]:
                    self.measurement_keys[m].append(p_)
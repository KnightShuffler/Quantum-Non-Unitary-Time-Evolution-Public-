import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

import os
import logging
from math import isclose
from itertools import combinations

from qnute.hamiltonian import Hamiltonian
from qnute.helpers import int_to_base
from qnute.helpers.lattice import get_center
from qnute.helpers.lattice import get_m_sphere
from qnute.helpers.lattice import min_bounding_sphere
from qnute.helpers.lattice import in_lattice
from qnute.helpers.lattice import manhattan_dist
from qnute.helpers.pauli import ext_domain_pauli
from qnute.helpers.pauli import pauli_string_prod
from qnute.helpers.pauli import odd_y_pauli_strings

class QNUTE_params:
    def __init__(self, H:Hamiltonian, lattice_dim, lattice_bound, qubit_map=None):
        # None qubit_map corresponds to a default 1D mapping
        if qubit_map == None:
            if lattice_dim != 1:
                raise ValueError('Default qubit map only available for 1D topology')
            self.qubit_map = {}
            for i in range(lattice_bound):
                self.qubit_map[(i,)] = i
        else:
            v = QNUTE_params.verify_map(lattice_dim, lattice_bound, qubit_map)
            if v != True:
                logging.ERROR('Qubit map not valid!')
                raise ValueError(v)
            self.qubit_map = qubit_map
        self.invert_map = {index:coord for coord,index in self.qubit_map.items()}
        self.nbits = len(self.qubit_map)

        self.lattice_dim = lattice_dim
        self.lattice_bound = lattice_bound

        self.H = H
        self.QNUTE_H = None

        self.odd_y_strings = {}

        # self.h_domains = [hm[2] for hm in self.H.hm_list]
        self.u_domains = []
        # self.mix_domains = []
        self.h_measurements = []
        self.u_measurements = []
        self.mix_measurements = []

        self.reduce_dimension_flag = None

        self.D = 0

        # QNUTE Run Parameters
        self.dt = 0.0
        self.delta = 0.0
        self.N = 0
        self.num_shots = 0

        self.backend = None
        self.init_sv = None
        self.init_circ = None

        self.circuit_flag = True
        self.store_state_vector = True
        self.taylor_norm_flag = False
        self.taylor_truncate_h = -1
        self.taylor_truncate_a = -1
        self.trotter_flag = False

        # Measurements to be made at the end of each QNUTE step
        self.objective_measurements = []

        # Logging Information
        self.log_path = ''
        self.fig_path = ''
        self.run_name = ''

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

    @staticmethod
    def get_new_domain(active, D, d, l, qubit_map)->set[int]:
        '''
        Returns a list of the extended/shrunk domain of a Hamiltonian 
        with domain size D on a d-dimensional lattice of bound l
        '''
        c = get_center(active)
        dom = get_m_sphere(c, D//2, d, l)
        return {qubit_map[coord] for coord in dom if coord in qubit_map}

    def load_measurement_keys(self, m, domain_ops):
        '''
        Calculates the measurement operators required for the m-th term 
        in the Hamiltonian's hm_list
        h_measurements - Pauli indices for the Hamiltonian terms, domain h_domains[m]
        u_measurements - Pauli indices for the Unitary terms, domain u_domains[m]
        mix_measurement - Pauli indices for the products of h and u terms, domain mix_domains[m]
        '''
        # h_c, h_R = min_bounding_sphere(self.h_domains[m])
        # u_domain = [self.qubit_map[coord] for coord in self.u_domains[m]]
        # u_c, u_R = min_bounding_sphere(u_domain)

        # Measurements for c: Pauli strings in hm
        self.h_measurements[m] = self.QNUTE_H.get_hm_pterms(m)['pauli_id']

        u_domain = [self.invert_map[index] for index in self.u_domains[m]]
        
        # Measurements for S: Products of Pauli strings on the unitary domain
        # All Pauli strings of can be expressed as a product of two 
        # Pauli strings with only odd number of Ys, so all the Pauli strings
        # acting on the unitary domain must be measured to build S
        self.u_measurements[m] = np.array(domain_ops,dtype=np.uint32)
        for i,p in enumerate(self.u_measurements[m]):
            self.u_measurements[m][i] = ext_domain_pauli(p, self.u_domains[m], list(range(self.nbits)))
        
        # Measurements for b: Products of Pauli strings on the unitary domain with 
        # the Pauli strings in hm        
        # Only need to do this calculation if the unitary domain is smaller than the
        # Hamiltonian term's domain, otherwise, all measurement operators were accounted
        # for when building S

        if self.QNUTE_H.get_hm_term_support(m).issubset(self.u_domains[m]):
            self.mix_measurements[m] = self.u_measurements[m]
        else:
            mix_pstrings = []
            for I in self.h_measurements[m]:
                for J in self.u_measurements[m]:
                    mix_pstrings.append(pauli_string_prod(I,J,self.nbits)[0])
            mix_pstrings = set(mix_pstrings)
            self.mix_measurements[m] = np.zeros(len(mix_pstrings), dtype=np.uint32)
            for i,pstring in enumerate(mix_pstrings):
                self.mix_measurements[m][i] = pstring  

    def load_hamiltonian_params(self, D: int, u_domains:list[set[int]],
                                reduce_dim: bool =False, 
                                load_measurements: bool=True):
        '''
        Performs the precalculations to run QNUTE
        D: Diameter of the unitary domains
        reduce_dim: Flag for whether to use reduced dimension linear system
        load_measurements: Flag for whether to calculate the required measurements
        '''
        logging.debug('Performing Hamiltonian precalculations...')
        # hm_list = self.H.hm_list
        # nterms = self.H.num_terms
        self.D = D
        self.reduce_dimension_flag = reduce_dim

        logging.debug('\tCalculating Unitary Domains...')
        # Calculate the domains of the unitaries simulating each term
        # for domain in self.h_domains:
        #     self.u_domains.append(QNUTE_params.get_new_domain(domain, D, self.lattice_dim, self.lattice_bound, self.qubit_map))
        #     self.mix_domains.append( list(set(domain) | set(self.u_domains[-1])) )
        self.u_domains = u_domains
        amplitude_splits = QNUTE_params.get_u_domain_amplitude_splits(self.H, u_domains, self.invert_map)
        self.QNUTE_H = self.H.rearrange_terms(u_domains, amplitude_splits)
        nterms = self.QNUTE_H.num_terms

        self.h_measurements = [0] * nterms
        self.u_measurements = [0] * nterms
        self.mix_measurements = [0] * nterms

        # Check if the terms are real
        if reduce_dim:
            logging.debug('\tCalculating Required Odd-Y Pauli Strings...')
            self.QNUTE_H.calculate_real_terms()

            # Initialize the keys for the odd y strings
            for m in range(nterms):
                if self.QNUTE_H.real_term_flags[m]:
                    self.odd_y_strings[len(self.u_domains[m])] = None
            
            # Load the odd Y Pauli Strings
            for y_len in self.odd_y_strings:
                self.odd_y_strings[y_len] = odd_y_pauli_strings(y_len)
        
        if load_measurements:
            logging.debug('  Calculating Required Pauli Measurements...')

            # Calculate the strings to measure for each term:
            for m in range(nterms):
                # Initialize list of keys for the m-th term
                self.h_measurements[m] = []
                self.u_measurements[m] = []
                self.mix_measurements[m] = []
                ndomain = len(self.u_domains[m])
                # Populate the keys
                # domain_ops = self.odd_y_strings[ndomain] if reduce_dim and self.QNUTE_H.real_term_flags[m] else list(range(4**ndomain))
                domain_ops = list(range(4**ndomain))
                self.load_measurement_keys(m, domain_ops)
    
    def set_run_params(self, dt, delta, N, num_shots, backend, init_circ=None,
                       init_sv=None, store_state_vector=True,
                       taylor_norm_flag=False, taylor_truncate_h=-1,
                       taylor_truncate_a=-1, trotter_flag=False, 
                       objective_meas_list=None):
        self.dt = dt
        self.delta = delta
        self.N = N
        self.num_shots = num_shots
        self.backend = backend
        self.store_state_vector = store_state_vector
        if not self.store_state_vector:
            raise ValueError('params.store_state_vector=False is not supported yet.')
        self.taylor_norm_flag = taylor_norm_flag
        self.taylor_truncate_h = taylor_truncate_h
        self.taylor_truncate_a = taylor_truncate_a
        self.trotter_flag = trotter_flag

        # Determine if the calculation is matrix based or QuantumCircuit based
        if backend is None:
            self.circuit_flag = False
        else:
            self.circuit_flag = True

        # Set the initializing circuit
        if init_circ is None:
            if init_sv is None:
                self.init_circ = QuantumCircuit(self.nbits)
                self.init_sv = Statevector.from_label('0'*self.nbits)
            else:
                if isinstance(init_sv, Statevector):
                    self.init_sv = init_sv
                else:
                    self.init_sv = Statevector(init_sv)
                self.init_circ = QuantumCircuit(self.nbits)
                self.init_circ.initialize(self.init_sv, list(range(self.nbits)))
        else:
            self.init_circ = init_circ
            # Raise an exception if the user inputted an initializing circuit in a simulation that doesn't use QuantumCircuits
            if not self.circuit_flag:
                raise ValueError('Provided init_circ instead of init_sv for a simulation that does not use QuantumCircuit')
        
        # Set the list of objective measurements to be made
        if objective_meas_list is None:
            self.objective_measurements = []
        else:
            for m_list in objective_meas_list:
                qbits = m_list[1]
                for p in m_list[0]:
                    pstring = int_to_base(p, 4, len(qbits))
                    m_name=''
                    for i in range(len(qbits)):
                        if pstring[i] == 0: m_name += 'I'
                        else: m_name += chr(ord('X')+pstring[i]-1)
                        m_name += '_'
                        m_name += str(qbits[i])
                        if i < len(qbits) - 1:
                            m_name += ' '
                    self.objective_measurements.append([m_name, p, qbits])


    def set_identifiers(self, log_path, fig_path, run_name):
        self.log_path = log_path
        self.fig_path = fig_path
        self.run_name = run_name

        run_id_string = 'db={:0.2f}/delta={:0.2f}/D={}/N={}/'.format(self.dt, self.delta, self.D, self.N)
        
        # Make sure the path name ends in /
        if log_path[-1] != '/':
            self.log_path += '/'
        if fig_path[-1] != '/':
            self.fig_path += '/'
        
        self.log_path += run_id_string
        self.fig_path += run_id_string

        if run_name[-1] != '-':
            self.run_name += '-'

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)

    @staticmethod
    def get_minimum_cover(support:set[int], u_domains:list[set[int]], invert_map:dict[int,tuple[int]]) -> list[int]:
        support_center = get_center([invert_map[index] for index in support])
        u_domain_centers:list[np.ndarray[float]] = [get_center([invert_map[index] for index in domain]) for domain in u_domains]

        r = 1
        if len(support) > len(u_domains[0]):
            r+=1
        found_flag = False
        min_dist = float('inf')
        u_index_list = []
        while r < len(u_domains) and not found_flag:
            for u_indices in combinations(range(len(u_domains)), r):
                u = [u_domains[i] for i in u_indices]
                x = set.union(*u)
                if support.issubset(x):
                    found_flag = True
                    dist = 0.0
                    for i in u_indices:
                        dist += manhattan_dist(support_center, u_domain_centers[i])
                    if isclose(dist,min_dist):
                        u_index_list += list(u_indices)
                    else:
                        if dist < min_dist:
                            min_dist = dist
                            u_index_list = list(u_indices)
            r += 1
        
        if not found_flag:
            u_index_list = list(range(len(u_domains)))
        else:
            r -= 1
        return u_index_list
    
    @staticmethod
    def get_u_domain_amplitude_splits(ham:Hamiltonian, u_domains:list[set[int]], invert_map:dict[int,tuple[int]]):
        amplitude_splits = np.zeros((ham.num_terms,len(u_domains)),dtype=np.float64)

        for term in range(ham.num_terms):
            support = ham.get_hm_term_support(term)
            
            u_index_list = QNUTE_params.get_minimum_cover(support, [set(dom) for dom in u_domains], invert_map)
            frac = 1.0/len(u_index_list)
            for ui in u_index_list:
                amplitude_splits[term,ui] += frac
        return amplitude_splits

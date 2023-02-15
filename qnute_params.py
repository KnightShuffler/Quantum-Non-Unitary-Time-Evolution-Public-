import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import AerError
import os

from hamiltonians import Hamiltonian
from helpers import *

class QNUTE_params:
    def __init__(self, Ham: Hamiltonian):
        self.H = Ham
        self.odd_y_strings = {}
        self.h_domains = []
        for hm in Ham.hm_list:
            self.h_domains.append(hm[2])
        self.u_domains = []
        self.mix_domains = []
        self.h_measurements = {}
        self.u_measurements = {}
        self.mix_measurements = {}

        self.reduce_dimension_flag = None

        self.small_u_domain_flags = [False] * Ham.num_terms

        self.nbits = Ham.nbits
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

    def get_new_domain(active, D, d, l):
        '''
        Returns a list of the extended/shrunk domain of a Hamiltonian 
        with domain size D on a d-dimensional lattice of bound l
        '''
        c = get_center(active)
        dom = get_m_sphere(c, D//2, d, l)
        if d == 1:
            domain = [ point[0] for point in dom ]
            return domain
        else:
            return dom

    def load_measurement_keys(self, m, domain_ops):
        '''
        Calculates the measurement operators required for the m-th term 
        in the Hamiltonian's hm_list
        h_measurements - Pauli indices for the Hamiltonian terms, domain h_domains[m]
        u_measurements - Pauli indices for the Unitary terms, domain u_domains[m]
        mix_measurement - Pauli indices for the products of h and u terms, domain mix_domains[m]
        '''
        hm = self.H.hm_list[m]
        h_c, h_R = min_bounding_sphere(hm[2])
        u_domain = self.u_domains[m]
        u_c, u_R = min_bounding_sphere(u_domain)

        # Measurements for c: Pauli strings in hm
        self.h_measurements[m] = hm[0]
        
        # Measurements for S: Products of Pauli strings on the unitary domain
        # All Pauli strings of can be expressed as a product of two 
        # Pauli strings with only odd number of Ys, so all the Pauli strings
        # acting on the unitary domain must be measured to build S
        self.u_measurements[m] = list(range(4**len(u_domain)))
        
        # Measurements for b: Products of Pauli strings on the unitary domain with 
        # the Pauli strings in hm

        # List of pauli dictionaries for the operators of the unitary's domain
        h_ops = [
            pauli_index_to_dict(hm[0][j], hm[2]) for j in range(len(hm[0]))
        ]

        def add_entry(meas, entry):
            for dict in meas:
                if same_pauli_dicts(entry, dict):
                    return
            meas.append(entry)

        # Only need to do this calculation if the unitary domain is smaller than the
        # Hamiltonian term's domain, otherwise, all measurement operators were accounted
        # for when building S
        if u_R < h_R:
            self.small_u_domain_flags[m] = True
            mix_dicts = []
            for i in domain_ops:
                i_dict = pauli_index_to_dict(i, u_domain)
                for j_dict in h_ops:
                    prod_dict, coeff = pauli_dict_product(i_dict, j_dict)
                    add_entry(mix_dicts, prod_dict)
            # Calculate indices for the mix_measurements
            for i in range(len(mix_dicts)):
                mix_dict = mix_dicts[i]
                pauli_id = 0
                power = 0
                for j in range(len(self.mix_domains[m])):
                    qbit = self.mix_domains[m][j]
                    if qbit in mix_dict.keys():
                        index = mix_dict[qbit]
                        pauli_id += index * 4**power
                    power += 1
                self.mix_measurements[m].append(pauli_id)

    def load_hamiltonian_params(self, D: int, reduce_dim: bool =False, 
                                load_measurements: bool=True):
        '''
        Performs the precalculations to run QNUTE
        D: Diameter of the unitary domains
        reduce_dim: Flag for whether to use reduced dimension linear system
        load_measurements: Flag for whether to calculate the required measurements
        '''
        print('Performing Hamiltonian precalculations...')
        hm_list = self.H.hm_list
        nterms = self.H.num_terms
        self.D = D
        self.reduce_dimension_flag = reduce_dim

        print('\tCalculating Unitary Domains...',end=' ',flush=True)
        # Calculate the domains of the unitaries simulating each term
        for hm in hm_list:
            self.u_domains.append(QNUTE_params.get_new_domain(hm[2], D, self.H.d, self.H.l))
            self.mix_domains.append( list(set(hm[2]) | set(self.u_domains[-1])) )
        print('Done')

        # Check if the terms are real
        if reduce_dim:
            print('\tCalculating Required Odd-Y Pauli Strings...', end=' ', flush=True)

            # Initialize the keys for the odd y strings
            for m in range(nterms):
                if self.H.real_term_flags[m]:
                    self.odd_y_strings[len(self.u_domains[m])] = None
            
            # Load the odd Y Pauli Strings
            for y_len in self.odd_y_strings.keys():
                self.odd_y_strings[y_len] = odd_y_pauli_strings(y_len)
            print('Done')
        
        if load_measurements:
            print('\tCalculating Required Pauli Measurements...', end=' ', flush=True)

            # Calculate the strings to measure for each term:
            for m in range(nterms):
                # Initialize list of keys for the m-th term
                self.h_measurements[m] = []
                self.u_measurements[m] = []
                self.mix_measurements[m] = []
                ndomain = len(self.u_domains[m])
                # Populate the keys
                domain_ops = self.odd_y_strings[ndomain] if reduce_dim and self.H.real_term_flags[m] else list(range(4**ndomain))
                self.load_measurement_keys(m, domain_ops)
        print('Done')
    
    def set_run_params(self, dt, delta, N, num_shots, 
    backend, init_circ=None, init_sv=None, store_state_vector=True,
    taylor_norm_flag=False, taylor_truncate_h=-1, taylor_truncate_a=-1, 
    trotter_flag=False,
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
        if objective_meas_list is not None:
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

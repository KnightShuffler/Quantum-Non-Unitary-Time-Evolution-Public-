import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import AerError
import os

from hamiltonians import Hamiltonian
from helpers import *

DRIFT_NONE = 0
DRIFT_A = 1
DRIFT_THETA_2PI = 2
DRIFT_THETA_PI_PI = 3

class QITE_params:
    def __init__(self, Ham: Hamiltonian):
        # Hamiltonian Information
        # self.hm_list = []
        # self.nterms = 0
        # self.real_term_flags = []
        self.H = Ham
        self.small_domain_flags = []
        self.odd_y_strings = {}
        self.u_domains = []
        self.measurement_keys = {}

        self.nbits = 0
        self.D = 0

        # QITE Run Parameters
        self.db = 0.0
        self.delta = 0.0
        self.N = 0
        self.num_shots = 0

        self.backend = None
        self.init_sv = None
        self.init_circ = None

        self.drift_type = DRIFT_NONE

        # GPU usage Flags
        self.gpu_simulator_flag = False
        self.gpu_calculation_flag = False

        # Logging Information
        self.log_path = ''
        self.fig_path = ''
        self.run_name = ''
    
    # def domain_size(qbits):
    #     return max(qbits) - min(qbits) + 1

    def get_new_domain(active, D, d, l):
        '''
        Returns a list of the extended/shrunk domain of a Hamiltonian 
        with domain size D on a d-dimensional lattice of bound l
        '''
        c = get_center(active)
        return get_m_sphere(c, D//2, d, l)
    
    def load_measurement_keys(self, m, domain_ops):
        '''
        Calculates the measurement operators required for the m-th term 
        in the Hamiltonian's hm_list
        '''
        hm = self.H.hm_list[m]
        c, R = min_bounding_sphere(hm[2])
        active = get_m_sphere(c, R, self.H.d, self.H.l)
        domain = self.u_domains[m]
        ndomain = len(domain)
        nactive = len(active)
        
        if ndomain >= nactive:
            big_domain = domain
            self.small_domain_flags[m] = False
        else:
            big_domain = active
            self.small_domain_flags[m] = True

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

    def load_hamiltonian_params(self, D):
        print('Loading Hamiltonian Parameters...',end=' ',flush=True)
        # Validate the passed parameters
        # if not hamiltonians.is_valid_domain(hm_list, D, nbits):
        #     raise ValueError('The domain size D is not valid for the hamiltonian, parameters not set.')

        hm_list = self.H.hm_list
        nterms = self.H.num_terms
        nbits = self.H.nbits

        self.D = D

        print('Calculating Unitary Domains...',end=' ',flush=True)
        # Calculate the domains of the unitaries simulating each term
        for hm in hm_list:
            self.u_domains.append(QITE_params.get_new_domain(hm[2], D, self.H.d, self.H.l))
        print('Done')

        # Check if the terms are real
        print('Calculating Required Odd-Y Pauli Strings...', end=' ', flush=True)

        # Initialize the keys for the odd y strings
        for m in range(self.nterms):
            if self.H.real_term_flags[m]:
                self.odd_y_strings[len(self.domains[m])] = None
        
        # Load the odd Y Pauli Strings
        for y_len in self.odd_y_strings.keys():
            self.odd_y_strings[y_len] = odd_y_pauli_strings(y_len)
        print('Done')
        
        print('Calculating Required Pauli Measurements...', end=' ', flush=True)
        self.small_domain_flags = [False] * self.nterms

        # Calculate the strings to measure for each term:
        for m in range(self.nterms):
            # Initialize list of keys for the m-th term
            self.measurement_keys[m] = []
            ndomain = len(self.domains[m])
            # Populate the keys
            domain_ops = self.odd_y_strings[ndomain] if self.real_term_flags[m] else list(range(4**ndomain))
            self.load_measurement_keys(m, domain_ops)
        print('Done')
    
    def set_run_params(self, db, delta, N, num_shots, 
    backend, init_circ=None, init_sv=None, drift_type=DRIFT_NONE,
    gpu_sim_flag=False, gpu_calc_flag=False):
        self.db = db
        self.delta = delta
        self.N = N
        self.num_shots = num_shots
        self.backend = backend
        self.drift_type = drift_type
        self.gpu_simulator_flag = gpu_sim_flag
        self.gpu_calc_flag = gpu_calc_flag

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
        
        if self.gpu_simulator_flag:
            try:
                self.backend.set_options(device='GPU')
            except AerError as e:
                print(e)
                print('Unable to set simulation device to GPU, proceeding with CPU simulation')
                self.gpu_simulator_flag = False

    def set_identifiers(self, log_path, fig_path, run_name):
        self.log_path = log_path
        self.fig_path = fig_path
        self.run_name = run_name

        run_id_string = 'db={:0.2f}/delta={:0.2f}/D={}/N={}/'.format(self.db, self.delta, self.D, self.N)
        
        # Make sure the path name ends in /
        if log_path[-1] != '/':
            self.log_path += '/'
        if fig_path[-1] != '/':
            self.fig_path += '/'
        
        self.log_path += run_id_string
        self.fig_path += run_id_string

        if run_name[-1] != '-':
            self.run_name += '-'
        
        # if self.gpu_simulator_flag:
        #     self.run_name += 'GPU-'

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)

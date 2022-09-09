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

        self.small_u_domain_flags = [False] * Ham.num_terms

        self.nbits = Ham.nbits
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

    def load_hamiltonian_params(self, D):
        print('Loading Hamiltonian Parameters...',end=' ',flush=True)
        # Validate the passed parameters
        # if not hamiltonians.is_valid_domain(hm_list, D, nbits):
        #     raise ValueError('The domain size D is not valid for the hamiltonian, parameters not set.')

        hm_list = self.H.hm_list
        nterms = self.H.num_terms

        self.D = D

        print('Calculating Unitary Domains...',end=' ',flush=True)
        # Calculate the domains of the unitaries simulating each term
        for hm in hm_list:
            self.u_domains.append(QITE_params.get_new_domain(hm[2], D, self.H.d, self.H.l))
            self.mix_domains.append( list(set(hm[2]) | set(self.u_domains[-1])) )
        print('Done')

        # Check if the terms are real
        print('Calculating Required Odd-Y Pauli Strings...', end=' ', flush=True)

        # Initialize the keys for the odd y strings
        for m in range(nterms):
            if self.H.real_term_flags[m]:
                self.odd_y_strings[len(self.u_domains[m])] = None
        
        # Load the odd Y Pauli Strings
        for y_len in self.odd_y_strings.keys():
            self.odd_y_strings[y_len] = odd_y_pauli_strings(y_len)
        print('Done')
        
        print('Calculating Required Pauli Measurements...', end=' ', flush=True)

        # Calculate the strings to measure for each term:
        for m in range(nterms):
            # Initialize list of keys for the m-th term
            self.h_measurements[m] = []
            self.u_measurements[m] = []
            self.mix_measurements[m] = []
            ndomain = len(self.u_domains[m])
            # Populate the keys
            domain_ops = self.odd_y_strings[ndomain] if self.H.real_term_flags[m] else list(range(4**ndomain))
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

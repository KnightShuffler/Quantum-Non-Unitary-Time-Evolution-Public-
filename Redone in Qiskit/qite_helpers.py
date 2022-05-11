import numpy as np
from helper import *
import hamiltonians

from os import path, makedirs
import matplotlib.pyplot as plt


from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def domain_size(qbits):
    '''
    Returns the size of the domain of a list of qbits assuming a linear topology
    '''
    return np.max(qbits) - np.min(qbits) + 1

def extended_domain(hm, D, nbits):
    '''
    Returns a list of the extended domain of a k-local Hamiltonian term hm
    with a domain size D, and nbits qubits in a linear topology
    
    Note: D must be greater than the range of the hamiltonian term
    '''
    if (domain_size(hm[2])) % 2 == 0:
        if D%2 == 1:
            D = D-1
        center = (np.min(hm[2]) + np.max(hm[2])) // 2
        radius = D//2
        _min = max( center+1 - radius ,0 )
        _max = min( center + radius, nbits-1 )
        return list(range(_min, _max+1))
    else:
        if D%2 == 0:
            D = D-1
        center = (np.min(hm[2]) + np.max(hm[2])) // 2
        radius = D//2
        _min = max( center-radius, 0 )
        _max = min( center+radius, nbits-1 )
        return list(range(_min,_max+1))

class qite_params:
    def __init__(self):
        # Set by initialize
        self.odd_y_flag = False
        self.small_d_flag = False
        self.odd_y_strings = []
        self.domains = []
        self.measurement_keys = []
        self.nbits = 0
        self.D = 0
        self.hm_list = []
        self.nterms = 0

        # set by set_run_params
        self.db = 0.0
        self.delta = 0.0
        self.N = 0
        self.backend = None
        self.num_shots = 0
        self.init_sv = None
        self.init_circ = None
        self.gpu_flag = False

        # set by set_identifiers
        self.log_path = ''
        self.fig_path = ''
        self.id = ''
    
    def initialize(self, hm_list, nbits, D):
        self.hm_list = hm_list
        self.nbits = nbits
        self.D = D
        
        # Check if it's a real hamiltonian in the Z-basis
        self.odd_y_flag = hamiltonians.is_real_hamiltonian(hm_list)

        # Check if the domain size is smaller than k
        self.small_d_flag = D < hamiltonians.get_k(hm_list)

        # Verify that the domain size is valid for the Hamiltonian
        if not hamiltonians.is_valid_domain(hm_list, D, nbits):
            raise ValueError('Invalid domain size D for the the Hamiltonian')

        # The number of terms in the Hamiltonian
        self.nterms = len(hm_list)
        
        # Save the domains on which each term of the hamiltonian is simulated
        # domains = []
        for i in range(self.nterms):
            self.domains.append(extended_domain(hm_list[i], D, nbits))

        # Save the Pauli strings with odd number of Ys for each domain size
        # odd_y_strings = []
        # Only populate if it's a real Hamiltonian
        if self.odd_y_flag:
            for i in range(self.nterms):
                self.odd_y_strings.append(odd_y_pauli_strings( len(self.domains[i] )))
        
        # Save which Pauli strings need to be measured for each term in the Hamiltonian
        # measurement_keys = []

        if not self.small_d_flag:
            # If D >= k, the vector a will have same dimensionality as the number of relevant
            # D-length pauli strings, so all measurements will be done on the domain of D-length
            # Pauli strings
            if not self.odd_y_flag:
                # Measure all D-length Pauli strings for each term
                for i in range(self.nterms):
                    self.measurement_keys.append(list(range( 4**len(self.domains[i]) )))
            else:
                for i in range(self.nterms):
                    self.measurement_keys.append([])
                    # Measure all D-length, odd-Y Pauli strings for each term
                    for j in self.odd_y_strings[i]:
                        for k in self.odd_y_strings[i]:
                            self.measurement_keys[-1].append(pauli_string_prod(j,k,len(self.domains[i]))[0])
                    # self.measurement_keys.append(self.odd_y_strings[i].copy())
                    # Also measure the products of the D-length, odd-Y Pauli strings with the 
                    # strings present in the hamiltonian term
                    for py in self.odd_y_strings[i]:
                        for p in hm_list[i][0]:
                            ext_p = ext_domain_pauli(p, hm_list[i][2], self.domains[i])
                            p_,c_ = pauli_string_prod(py, ext_p, len(self.domains[i]))
                            if p_ not in self.measurement_keys[-1]:
                                self.measurement_keys[-1].append(p_)
                    # Include the measurements of the Pauli strings present in the hamiltonian term
                    for p in hm_list[i][0]:
                        ext_p = ext_domain_pauli(p, hm_list[i][2], self.domains[i])
                        if ext_p not in self.measurement_keys[-1]:
                            self.measurement_keys[-1].append(ext_p)
        else:
            # If D < k, the vector a will also have same dimensionality as the number of 
            # revelant D-length Pauli strings, but since the domain is smaller than the 
            # original, we also need to consider a few k-length Pauli strings due to the 
            # product terms.
            if not self.odd_y_flag:
                for i in range(self.nterms):
                    # Measure all the D-length Pauli strings
                    self.measurement_keys.append(list(range( 4**len( self.domains[i] ))))

                    # Calculate the full domain of the term
                    full_domain = list(range( np.min(hm_list[i][2]), np.max(hm_list[i][2])+1 ))
                    # Measure the k-length Pauli string IDs for the products of all the above 
                    # strings with the strings present in the hamiltonian term
                    if len(self.domains[i]) < len(full_domain):
                        for d_id in range(4**len(self.domains[i])):
                            ext_d_id = ext_domain_pauli(d_id, self.domains[i], full_domain)
                            for p in hm_list[i][0]:
                                ext_p_id = ext_domain_pauli(p, hm_list[i][2], full_domain)
                                # Save in the form of a tuple (pauli ID 1, pauli ID 2, number of qubits)
                                self.measurement_keys[-1].append( (ext_d_id, ext_p_id, len(full_domain)) )
            else:
                for i in range(self.nterms):
                    # Measure all the D-length, odd-Y Pauli strings
                    # self.measurement_keys.append( self.odd_y_strings[i].copy() )
                    self.measurement_keys.append([])
                    # Measure all D-length, odd-Y Pauli strings for each term
                    for j in self.odd_y_strings[i]:
                        for k in self.odd_y_strings[i]:
                            self.measurement_keys[-1].append(pauli_string_prod(j,k,len(self.domains[i]))[0])

                    # Calculate the full domain of the term
                    full_domain = list(range( np.min(hm_list[i][2]), np.max(hm_list[i][2])+1 ))
                    # Measure the k-length Pauli string IDs for the products of all the above 
                    # strings with the strings present in the hamiltonian term
                    if len(self.domains[i]) < len(full_domain):
                        for d_id in self.odd_y_strings[i]:
                            ext_d_id = ext_domain_pauli(d_id, self.domains[i], full_domain)
                            for p in hm_list[i][0]:
                                ext_p_id = ext_domain_pauli(p, hm_list[i][2], full_domain)
                                # Save in the form of a tuple (pauli ID 1, pauli ID 2, number of qubits)
                                self.measurement_keys[-1].append( (ext_d_id, ext_p_id, len(full_domain)) )
                        # Include measurements for all of the pauli strings in hm (already accounted for in the other case)
                        for p in hm_list[i][0]:
                            ext_p_id = ext_domain_pauli(p, hm_list[i][2], full_domain)
                            self.measurement_keys[-1].append( (0, ext_p_id, len(full_domain)) )

                    else:
                        for py in self.odd_y_strings[i]:
                            for p in hm_list[i][0]:
                                ext_p = ext_domain_pauli(p, hm_list[i][2], self.domains[i])
                                p_,c_ = pauli_string_prod(py, ext_p, len(self.domains[i]))
                                if p_ not in self.measurement_keys[-1]:
                                    self.measurement_keys[-1].append(p_)
                        # Include measurements for all of the pauli strings in hm (already accounted for in the other case)
                        for p in hm_list[i][0]:
                            ext_p = ext_domain_pauli(p, hm_list[i][2], self.domains[i])
                            if ext_p not in self.measurement_keys[-1]:
                                self.measurement_keys[-1].append(ext_p)
  
    def set_run_params(self, db, delta, N, num_shots, backend, init_sv, init_circ=None, gpu_flag=False):
        self.db = db
        self.delta = delta
        self.N = N
        self.num_shots = num_shots
        self.backend = backend
        self.gpu_flag = gpu_flag

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
            
    def set_identifiers(self, log_path, fig_path, id):
        self.log_path = log_path
        self.fig_path = fig_path
        
        # Make sure the path name ends in /
        if log_path[-1] != '/':
            self.log_path += '/'
        if fig_path[-1] != '/':
            self.fig_path += '/'

        self.id = id
        if id[-1] != '-':
            self.id += '-'

        if not path.exists(self.log_path):
            makedirs(self.log_path)
        if not path.exists(self.fig_path):
            makedirs(self.fig_path)

def plot_data(fig_title, run_id, params, E, statevectors, eig_flag, prob_flag):
    plt.clf()

    if prob_flag:
        fig,axs = plt.subplots(1,2, figsize=(12,5), sharex=True)
        energy_plot = axs[0]
        prob_plot = axs[1]

        energy_plot.set_title('Mean Energy in QITE')
        prob_plot.set_title('Ground State Probability in QITE')
    else:
        fig,axs = plt.subplots(1,1,figsize=(6,5))
        energy_plot = axs
    
    fig.suptitle(fig_title, fontsize=16)
    plt.subplots_adjust(top=0.85)
    
    energy_plot.plot(np.arange(params.N+1)*params.db, E, 'ro-', label='Mean Energy of State')
    if eig_flag:
        w,v = hamiltonians.get_spectrum(params.hm_list, params.nbits)
        for eig in w:
            eig_line = energy_plot.axhline(y=eig.real, color='k', linestyle='--')
        eig_line.set_label('Hamiltonian Energy Levels')
    
    energy_plot.set_xlabel('Imaginary Time')
    energy_plot.set_ylabel('Energy')
    energy_plot.grid()
    energy_plot.legend(loc='best')

    if prob_flag:
        if not eig_flag:
            w,v = hamiltonians.get_spectrum(params.hm_list, params.nbits)
        w_sort_i = np.argsort(w)

        gs_probs = np.zeros(params.N+1, dtype=float)

        for k in range(len(w)):
            i = w_sort_i[k]
            if k == 0:
                prev_i = i
            else:
                prev_i = w_sort_i[k-1]
            # stop looping if the energy increases from the ground state
            if w[i] > w[prev_i]:
                break
            
            vec = v[:,i]
            for j in range(params.N+1):
                gs_probs[j] += np.abs( np.vdot(vec, statevectors[j]) )**2
        prob_plot.plot(np.arange(params.N+1)*params.db, gs_probs, 'bs-')
        prob_plot.set_ylim([0.0, 1.0])
        prob_plot.grid()

    fig.tight_layout()

    plt.savefig(params.fig_path+params.id+run_id+'.png')
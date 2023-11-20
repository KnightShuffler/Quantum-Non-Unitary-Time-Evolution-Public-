import itertools
import numpy as np
from qnute.helpers.lattice import get_k_local_domains_in_map
from qnute.helpers.sampling import generate_random_complex_vector
from qnute.helpers.pauli import pauli_str_to_index
from qnute.hamiltonian import Hamiltonian
from .topology import TOPOLOGIES_DICT

def generate_random_hamiltonians(topology_id, k_local, num_hamiltonians=1000) -> (np.array, np.dtype):
    (num_qubits, lattice_dim, qubit_map) = TOPOLOGIES_DICT[topology_id]
    lattice_bound = np.max(list(qubit_map.values())) + 1
    (term_domains, num_terms, num_pauli_terms) = get_k_local_domains_in_map(k_local, lattice_dim, lattice_bound, qubit_map)
    
    TYPE_hamiltonian = np.dtype([('Pauli_ID_list', 'S'+str(k_local), (3**k_local,)),
                                 ('Amplitude_list', 'c16', (3**k_local,)),
                                 ('qubit_ID_list', 'i4', (k_local,))
                                ])
    hamiltonians = np.zeros(dtype=TYPE_hamiltonian, shape=(num_hamiltonians,num_terms))
    for n in range(num_hamiltonians):
        ham = hamiltonians[n]
        ham['qubit_ID_list'] = term_domains
        amps = generate_random_complex_vector(num_pauli_terms)
        j = 0
        for (_id,dom) in enumerate(term_domains):
            active_qubits = k_local - dom.count(-1)
            # TODO: Update to count for terms with I for k>2
            for (i,p) in enumerate(itertools.product('XYZ', repeat=active_qubits)):
                ham['Pauli_ID_list'][_id][i] = ''.join(p)
                ham['Amplitude_list'][_id][i] = amps[j]
                j += 1
    return hamiltonians, TYPE_hamiltonian

def ham_dtype_to_Hamiltonian(ham, qubit_map, lattice_dim, lattice_bound):
    hm_list = []
    for term in range(len(ham)):
        ham_term = [ [], [], [] ]
        for q in ham['qubit_ID_list'][term]:
            if (q != -1):
                ham_term[-1].append(qubit_map[q])
        for (i, p_str) in enumerate(ham['Pauli_ID_list'][term]):
            if p_str == b'':
                break
            ham_term[0].append(pauli_str_to_index(p_str))
            ham_term[1].append(ham['Amplitude_list'][term][i])
        hm_list.append(ham_term)
    invert_map = {v:k for (k,v) in qubit_map.items()}
    return Hamiltonian(hm_list, lattice_dim, lattice_bound, invert_map)

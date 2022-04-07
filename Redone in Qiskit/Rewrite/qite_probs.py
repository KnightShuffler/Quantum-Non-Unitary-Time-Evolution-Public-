import numpy as np
from helper import *
from qite import propogate, update_alist, log_qite_run

from qiskit import QuantumCircuit, Aer

import time

import matplotlib.pyplot as plt

from os import path, makedirs

from qiskit.quantum_info import Statevector
sv_sim = Aer.get_backend('statevector_simulator')

def pauli_expectation(psi, p, qbits):
    '''
    returns the theoretical expectation <psi|P|psi> where P is the pauli string acting on qbits, indexed by p
    '''
    nbits = int(np.log2(len(psi.data)))
    pstring = int_to_base(p,4,len(qbits))
    qc = QuantumCircuit(nbits)
    for i in range(len(qbits)):
        if pstring[i] == 1:
            qc.x(qbits[i])
        elif pstring[i] == 2:
            qc.y(qbits[i])
        elif pstring[i] == 3:
            qc.z(qbits[i])
    
    phi = psi.evolve(qc)
    
    return np.real(np.vdot(psi.data, phi.data))

def ideal_energy(alist, nbits, hm_list):
    Energy = 0
    qc = QuantumCircuit(nbits)
    propogate(qc, alist)
    init_sv = [0]*(2**nbits)
    init_sv[0] = 1
    psi = Statevector(init_sv).evolve(qc)
    for hm in hm_list:
        for j in range(len(hm[0])):
            Energy += hm[1][j] * pauli_expectation(psi, hm[0][j], hm[2])
    return Energy

def get_psis(hm_list, alist, nbits):
    nterms = len(hm_list)
    N = len(alist)//nterms
        
    states = []
    states.append(Statevector.from_label('0'*nbits))
    
    for i in range(1,N+1):
        qc = QuantumCircuit(nbits)
        propogate(qc, alist[0:i*nterms])
        states.append(states[0].evolve(qc))
    return states


def make_prob_plot(log_path, log_file, fig_path, fig_name):
    x = np.load(log_path+log_file,allow_pickle=True)
    db = x[0]
    delta = x[1]
    N = x[2]
    nbits = x[3]
    hm_list = x[4]
    alist = x[5]

    psis = get_psis(hm_list, alist, nbits)

    w,v = get_spectrum(hm_list,nbits)

    energy_state_probs = np.zeros([len(w), N+1], dtype=float)

    mean_energies = np.zeros(N+1,dtype=float)

    hmat = get_h_matrix(hm_list, nbits)

    for i in range(N+1):
        for j in range(len(w)):
            energy_state_probs[j][i] = np.abs( np.vdot(v[:,j], psis[i].data) )**2
        mean_energies[i] = np.real(np.conj(psis[i].data).T @ hmat @ psis[i].data)
    
    fig, axs = plt.subplots(1,2, figsize = (12,5))

    w = np.real(w)
    w_sort_i = sorted(range(len(w)), key=lambda k: w[k])

    # plt.clf()

    # fig.suptitle('Ideal QITE behaviour', fontsize=16)
    # plt.subplots_adjust(top=0.85)

    p1, = axs[0].plot(np.arange(0,N+1)*db, mean_energies, 'ro-')
    for ei in w:
        p2 = axs[0].axhline(y=ei, color='k', linestyle='--')
    axs[0].set_title('Mean Energy in QITE')
    axs[0].set_ylabel('Energy')
    axs[0].set_xlabel('Imaginary Time')
    axs[0].grid()
    axs[0].legend((p1,p2), ('Mean Energy of State', 'Hamiltonian Energy Levels'), loc='best')

    axs[1].plot(np.arange(0,N+1)*db, energy_state_probs[w_sort_i[0]], 'ro-', label='{:0.3f} energy eigenstate'.format(w[w_sort_i[0]]))
    axs[1].plot(np.arange(0,N+1)*db, energy_state_probs[w_sort_i[1]], 'g.-', label='{:0.3f} energy eigenstate'.format(w[w_sort_i[1]]))
    axs[1].plot(np.arange(0,N+1)*db, energy_state_probs[w_sort_i[2]], 'bs-', label='{:0.3f} energy eigenstate'.format(w[w_sort_i[2]]))
    axs[1].plot(np.arange(0,N+1)*db, energy_state_probs[w_sort_i[3]], 'kP-', label='{:0.3f} energy eigenstate'.format(w[w_sort_i[3]]))

    axs[1].set_title('Energy Eigenstate Probabilities in QITE')
    axs[1].set_ylim([0,1])
    axs[1].set_ylabel('Probability')
    axs[1].set_xlabel('Imaginary Time')

    axs[1].grid()
    axs[1].legend(loc='best')

    fig.tight_layout()

    if not(path.exists(fig_path)):
        makedirs(fig_path)
    plt.savefig(fig_path+fig_name)

if __name__ == '__main__':
    log_path = './qite_logs/shots=1000/'
    fig_path = './figs/probabilities/shots=1000/'

    for i in range(11,18+1):
        print('Iteration {} of 8'.format(i) )
        identifier = 'run'+'{:0>3}'.format(i)
        make_prob_plot(log_path, identifier+'.npy', fig_path, identifier)

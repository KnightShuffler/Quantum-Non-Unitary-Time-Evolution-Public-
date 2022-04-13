import numpy as np

from ideal_qite import ideal_qite
from helper import get_spectrum, get_h_matrix

from qiskit.quantum_info import Statevector

import matplotlib.pyplot as plt

from os import path, makedirs

# QITE Parameters
db = 0.05       # Size of imaginary time step
N = 30          # Number of imaginary time steps
shots = 1000    # Number of measurements taken for each circuit
delta = 0.1     # Regularizer value

# Hamiltonian Description
nbits = 2       # Number of qubits in the full system

hm_list = []
hm = [ [3+1*4], [np.sqrt(0.5)], [0,1] ]
hm_list.append(hm)
hm = [ [1,3], [0.5, 0.5], [1]]
hm_list.append(hm)
# For this example, the Hamiltonian is of the form: 1/sqrt(2) (Z_0 X_1) + 1/sqrt(2) (I_0 H_1)

log_path = './qite_logs/ideal_qite/'
fig_path = './figs/ideal_qite/'

if not(path.exists(log_path)):
    makedirs(log_path)
if not(path.exists(fig_path)):
    makedirs(fig_path)

# Get the spectrum of H
w,v = get_spectrum(hm_list,nbits)
w = np.real(w)
w_sort_i = sorted(range(len(w)), key=lambda k: w[k])

for gs_prob in np.arange(0, 11)*0.1:
    print('Iteration: gs_prob = {:0.1f}'.format(gs_prob))
    id = 'ideal-qite-{:0.1f}_gs_prob'.format(gs_prob)
    log_file = log_path + id
    fig_file = fig_path + id

    init_sv = Statevector( np.sqrt(gs_prob) * v[w_sort_i[0]] + 
                        np.sqrt(1.0-gs_prob) * v[w_sort_i[1]] )

    E,times,states = ideal_qite(db,delta,N,nbits,hm_list,init_sv,details=True,log=True,log_file=log_file)
    
    plt.clf()

    fig, axs = plt.subplots(1,2,figsize=(12,5))

    energy_state_probs = np.zeros([len(w), N+1], dtype=float)

    hmat = get_h_matrix(hm_list, nbits)

    for i in range(N+1):
        for j in range(len(w)):
            energy_state_probs[j][i] = np.abs( np.vdot(v[:,j], states[i].data) )**2

    fig.suptitle('Ideal QITE behaviour', fontsize=16)
    plt.subplots_adjust(top=0.85)

    p1, = axs[0].plot(np.arange(0,N+1)*db, E, 'ro-')
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

    plt.savefig(fig_file+'.png')
import numpy as np

from ideal_qite import ideal_qite
from ideal_qite_2nd_ord import ideal_qite_2nd_ord
from helper import get_spectrum, get_h_matrix, int_to_base

from qiskit.quantum_info import Statevector

import matplotlib.pyplot as plt

from os import path, makedirs

# QITE Parameters
db = 0.1       # Size of imaginary time step
N = 60          # Number of imaginary time steps
shots = 1000    # Number of measurements taken for each circuit
delta = 0.1     # Regularizer value

# Set true if using 2nd order trotterization
_2nd_ord_flag = False

# Hamiltonian Description
nbits = 2       # Number of qubits in the full system

hm_list = []

h_name = '1D AFM Transverse Ising - {} qubits, '.format(nbits)
# Hamiltonian of the form J sum<i,j>(Z_i Z_j) + h sum_i (X_i)
J = 1
h = 0.5
for i in range(nbits-1):
    hm = [ [3+4*3], [J], [i,i+1] ]
    hm_list.append(hm)
for i in range(nbits):
    hm = [ [1], [h], [i] ]
    hm_list.append(hm)
h_name += ' J={:0.2f}, h={:0.2f}'.format(J,h)

log_path = './qite_logs/ideal_qite{}/{}/db={:0.1f}/N={}/'.format('_2nd_ord' if _2nd_ord_flag else '',h_name,db,N)
fig_path = './figs/ideal_qite{}/{}/db={:0.1f}/N={}/'.format('_2nd_ord' if _2nd_ord_flag else '',h_name,db,N)

if not(path.exists(log_path)):
    makedirs(log_path)
if not(path.exists(fig_path)):
    makedirs(fig_path)

# Get the spectrum of H
w,v = get_spectrum(hm_list,nbits)
w = np.real(w)
w_sort_i = sorted(range(len(w)), key=lambda k: w[k])

# For plotting lines
colors = ['r','g','b','k','m']
markers = ['.','s','x','+','*','h','H','D','o']

def plot_data(E,states, prob_string, fig_file):
    plt.clf()

    fig, axs = plt.subplots(1,2,figsize=(12,5))

    energy_state_probs = np.zeros([len(w), N+1], dtype=float)

    for i in range(N+1):
        for j in range(len(w)):
            energy_state_probs[j][i] = np.abs( np.vdot(v[:,j], states[i].data) )**2

    fig.suptitle('Ideal QITE behaviour ({})\n{}'.format(h_name,prob_string), fontsize=16)
    plt.subplots_adjust(top=0.85)

    p1, = axs[0].plot(np.arange(0,N+1)*db, E, 'ro-')
    for ei in w:
        p2 = axs[0].axhline(y=ei, color='k', linestyle='--')
    axs[0].set_title('Mean Energy in QITE')
    axs[0].set_ylabel('Energy')
    axs[0].set_xlabel('Imaginary Time')
    axs[0].grid()
    axs[0].legend((p1,p2), ('Mean Energy of State', 'Hamiltonian Energy Levels'), loc='best')

    for j in range(len(w)):
        axs[1].plot(np.arange(N+1)*db, energy_state_probs[w_sort_i[j]], '{}{}-'.format(colors[j%len(colors)], markers[j%len(markers)]), label='{:0.3f} energy eigenstate'.format(w[w_sort_i[j]]))
    # axs[1].plot(np.arange(0,N+1)*db, energy_state_probs[w_sort_i[0]], 'ro-', label='{:0.3f} energy eigenstate'.format(w[w_sort_i[0]]))
    # axs[1].plot(np.arange(0,N+1)*db, energy_state_probs[w_sort_i[1]], 'g.-', label='{:0.3f} energy eigenstate'.format(w[w_sort_i[1]]))
    # axs[1].plot(np.arange(0,N+1)*db, energy_state_probs[w_sort_i[2]], 'bs-', label='{:0.3f} energy eigenstate'.format(w[w_sort_i[2]]))
    # axs[1].plot(np.arange(0,N+1)*db, energy_state_probs[w_sort_i[3]], 'kP-', label='{:0.3f} energy eigenstate'.format(w[w_sort_i[3]]))

    axs[1].set_title('Energy Eigenstate Probabilities in QITE')
    axs[1].set_ylim([0,1])
    axs[1].set_ylabel('Probability')
    axs[1].set_xlabel('Imaginary Time')

    axs[1].grid()
    axs[1].legend(loc='best')

    fig.tight_layout()

    plt.savefig(fig_file+'.png')

def multiple_runs(num_states, prob_incr):
    b = int(np.floor(1.0/prob_incr)) + 1
    for iter in range(b**(num_states-1)):
        init_sv = np.zeros(2**nbits,dtype=complex)
        probs = np.zeros(num_states)
        prob_string = ''
        digits = int_to_base(iter,b,num_states-1)
        for j in range(num_states-1):
            probs[j] = digits[j]*prob_incr
            prob_string += 'p{}={:0.2f},'.format(j, probs[j])
            init_sv += np.sqrt(probs[j]) * v[:,w_sort_i[j]]
        if np.sum(probs) > 1.0:
            continue
        else:
            probs[-1] = 1.0 - np.sum(probs)
            prob_string += 'p{}={:0.2f}'.format(num_states-1, probs[-1])
            init_sv += np.sqrt(probs[-1]) * v[:,w_sort_i[num_states-1]]
        
        init_sv = Statevector(init_sv)

        print('Iteration:',prob_string)
        log_file = log_path + prob_string
        fig_file = fig_path + prob_string

        if _2nd_ord_flag:
            E,times,states = ideal_qite_2nd_ord(db,delta,N,nbits,hm_list,init_sv,details=True,log=True,log_file=log_file)
        else:
            E,times,states = ideal_qite(db,delta,N,nbits,hm_list,init_sv,details=True,log=True,log_file=log_file)

        plot_data(E,states, prob_string, fig_file)

def single_run_probs(probs):
    if len(probs) > len(w):
        print('Error: The probs vector should have at most elements as the number of eigenvectors')
        return
    if np.abs(np.sum(probs) - 1.0) > 1e-5:
        print('Error: The probabilities should sum to 1, check the probabilities')
        return
    init_sv = np.zeros(2**nbits,dtype=complex)
    prob_string = ''
    for i in range(len(probs)):
        init_sv += np.sqrt(probs[i]) * v[:,w_sort_i[i]]
        prob_string += 'p{}={:0.2f},'.format(i, probs[i])
    
    init_sv = Statevector(init_sv)

    log_file = log_path + prob_string
    fig_file = fig_path + prob_string

    if _2nd_ord_flag:
        E,times,states = ideal_qite_2nd_ord(db,delta,N,nbits,hm_list,init_sv,details=True,log=True,log_file=log_file)
    else:
        E,times,states = ideal_qite(db,delta,N,nbits,hm_list,init_sv,details=True,log=True,log_file=log_file)

    plot_data(E,states, prob_string, fig_file)

def single_run_sv(sv, prob_string):
    log_file = log_path + prob_string
    fig_file = fig_path + prob_string

    init_sv = Statevector(sv)

    if _2nd_ord_flag:
        E,times,states = ideal_qite_2nd_ord(db,delta,N,nbits,hm_list,init_sv,details=True,log=True,log_file=log_file)
    else:
        E,times,states = ideal_qite(db,delta,N,nbits,hm_list,init_sv,details=True,log=True,log_file=log_file)
    
    plot_data(E,states,prob_string, fig_file)
        

# multiple_runs(2,0.5,_2nd_ord_flag)
# single_run([0.0, 0.0, 1.0, 0.0], _2nd_ord_flag)
single_run_sv([ np.sqrt(0.5)**nbits ]*(2**nbits), 'maximally mixed state')
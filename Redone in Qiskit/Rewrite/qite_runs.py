import numpy as np
from qite import qite
from y_qite import y_qite
from qite import aer_sim as backend
from helper import get_spectrum, is_real_hamiltonian

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


n_runs = 4          # Number of runs
run_offset = 0      # So as not to overwrite previous data

log_path = './qite_logs/odd_y/shots={}/'.format(shots)
fig_path = './figs/energies/odd_y/shots={}/'.format(shots)
run_identifier = 'run'

if not(path.exists(log_path)):
    makedirs(log_path)
if not(path.exists(fig_path)):
    makedirs(fig_path)

real_h_flag = is_real_hamiltonian(hm_list)

for run in range(n_runs):
    print('Running iteration {} of {}:'.format(run+1, n_runs))
    if real_h_flag:
        E,times = y_qite(db, delta, N, nbits, hm_list, backend, shots, details=True, log=True, log_file=log_path+run_identifier+'{:0>3}'.format(run+run_offset+1))
    else:
        E,times = qite(db, delta, N, nbits, hm_list, backend, shots, details=True, log=True, log_file=log_path+run_identifier+'{:0>3}'.format(run+run_offset+1))

    plt.clf()

    p1, = plt.plot(np.arange(0,N+1)*db, E, 'ro-')

    w,v = get_spectrum(hm_list, nbits)
    for energy_level in w:
        p2 = plt.axhline(y=energy_level, color='k', linestyle='--')
    
    plt.legend((p1,p2), ('Mean Energy of State', 'Hamiltonian Energy Levels'), loc='upper center')

    plt.xlabel('Imaginary Time')
    plt.ylabel('Energy')
    plt.grid()

    plt.savefig(fig_path + run_identifier + '{:0>3}'.format(run+run_offset+1))

    print('Run time = {:.2f} minutes\n'.format(np.sum(times)/60))

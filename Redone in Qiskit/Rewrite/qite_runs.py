import numpy as np
from qite import qite
from qite import aer_sim as backend
from helper import get_spectrum

import matplotlib.pyplot as plt

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


n_runs = 1          # Number of runs
run_offset = 0      # So as not to overwrite previous data

log_path = './qite_logs/'
fig_path = './figs/energies/'

for run in range(n_runs):
    print('Running iteration {} of {}:'.format(run+1, n_runs))
    E,times = qite(db, delta, N, nbits, hm_list, backend, shots, details=True, log=True, log_file=log_path+'run{:0>3}'.format(run+run_offset+1))

    p1, = plt.plot(np.arange(0,N+1)*db, E, 'ro-')

    w,v = get_spectrum(hm_list, nbits)
    for energy_level in w:
        p2 = plt.axhline(y=energy_level, color='k', linestyle='--')
    
    plt.legend((p1,p2), ('Mean Energy of State', 'Hamiltonian Energy Levels'), loc='upper center')

    plt.xlabel('Imaginary Time')
    plt.ylabel('Energy')
    plt.grid()

    plt.savefig(fig_path + 'run{:0>3}'.format(run+run_offset+1))

    print('Run time = {:.2f}\n'.format(np.sum(times)/60))

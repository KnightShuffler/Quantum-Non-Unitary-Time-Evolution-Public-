import numpy as np

import matplotlib.pyplot as plt

from ideal_qite import qite, sv_sim
from qite_helpers import qite_params, plot_data, log_data
import hamiltonians

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import AerError

###############################
#       QITE Parameters       #
# update these per your needs #
###############################

# Size of imaginary time step
db = 0.1
# Size of regularizer
delta = 0.1
# Number of iterations
N = 30

# Number of qubits in the system
nbits = 6
# Maximum domain size for the simulated unitaries
D = 4

# Hamiltonian Description
J = [1,1,1]
B = 0
hm_list = hamiltonians.short_range_heisenberg(nbits, J, B)
h_name = 'Short Range Heisenberg - {} qubits'.format(nbits)
h_params = 'J=[{:0.2f},{:0.2f},{:0.2f}], B={}'.format(J[0],J[1],J[2],B)

# Initial State of the run
init_sv = Statevector.from_label('01'*(nbits//2))
init_circ = None

# GPU Usage Flags:
gpu_solver_flag = False    # True if you want to solve the systems of linear equations with a GPU, uses cupy as a backend
gpu_simulator_flag = False # True if you want to simulate the quantum circuits with a GPU, uses qiskit-aer-gpu

if gpu_simulator_flag:
    try:
        sv_sim.set_options(device='GPU')
    except AerError as e:
        print(e)
        print('Quantum Circuit Simulation will use the CPU')

# Set the logging paths
param_path = '{}/{}/D={}/N={}/db={}/delta={}/'.format(h_name, h_params,D,N,db,delta)
log_path = './qite_logs/ideal_qite/' + param_path
fig_path = './figs/ideal_qite/' + param_path
run_name = 'run'
run_id = '001'

params = qite_params()
params.initialize(hm_list, nbits, D)
params.set_run_params(db, delta, N, 0, None, init_sv, init_circ, gpu_solver_flag)
params.set_identifiers(log_path,fig_path,run_name)

# Run Flags
time_flag = True # True if you want to log the iteration times

# Plotting Flags
#   Note: calculating the spectrum of the Hamiltonian involves converting it to a matrix
#   of dimension 2**nbits, enable these flags accordingly
eig_flag = False  # True if you want to plot the energy levels of the Hamiltonian
prob_flag = True # True if you want to plot the ground state probability during the run

# Logging Flags


E,times,statevectors,alist = qite(params, time_flag)

plot_data('{}\n{}'.format(h_name,h_params), run_id, params, E, statevectors, eig_flag, prob_flag)
log_data('{}-{}'.format(run_name, run_id), params, E, times, alist)
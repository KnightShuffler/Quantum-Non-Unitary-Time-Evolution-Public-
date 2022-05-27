import numpy as np

import matplotlib.pyplot as plt

from ideal_qite import qite, CP_IMPORT_FLAG
from qite_params import QITE_params
import hamiltonians
from log_data import log_data, plot_data

from qiskit import QuantumCircuit, Aer
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
B = 1
hm_list = hamiltonians.short_range_heisenberg(nbits, J, B)
h_name = 'Short Range Heisenberg - {} qubits'.format(nbits)
h_params = 'J=[{:0.2f},{:0.2f},{:0.2f}], B={}'.format(J[0],J[1],J[2],B)

# Initial State of the run
init_sv = Statevector.from_label('01'*(nbits//2))
init_circ = None

# GPU Usage Flags:
gpu_solver_flag = False    # True if you want to solve the systems of linear equations with a GPU, uses cupy as a backend
gpu_simulator_flag = False # True if you want to simulate the quantum circuits with a GPU, uses qiskit-aer-gpu

# Set the logging paths
param_path = '{}/{}/'.format(h_name, h_params)
log_path = './qite_logs/ideal_qite/' + param_path
fig_path = './figs/ideal_qite/' + param_path
run_name = 'run'
run_id = '001'

params = QITE_params()
params.load_hamiltonian_params(hm_list, nbits, D)
params.set_run_params(db, delta, N, 0, 
Aer.get_backend('statevector_simulator'), init_circ, init_sv, 
gpu_simulator_flag, gpu_solver_flag and CP_IMPORT_FLAG)
params.set_identifiers(log_path,fig_path,run_name)

# Run Flags
time_flag = True # True if you want to log the iteration times

# Plotting Flags
#   Note: calculating the spectrum of the Hamiltonian involves converting it to a matrix
#   of dimension 2**nbits, enable these flags accordingly
gs_flag = True   # True if you want to plot the ground state energy of the Hamiltonian
prob_flag = False # True if you want to plot the ground state probability during the run

# Logging Flags


E,times,statevectors,alist = qite(params)

plot_data('{}\n{}'.format(h_name,h_params), run_id, params, E, statevectors, gs_flag, prob_flag)
log_data('{}-{}'.format(run_name, run_id), params, E, times, alist)
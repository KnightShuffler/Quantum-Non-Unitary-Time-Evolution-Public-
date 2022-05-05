import numpy as np

import matplotlib.pyplot as plt

from ideal_qite import qite
from qite_helpers import qite_params, plot_data, log_data
import hamiltonians

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

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
nbits = 4
# Maximum domain size for the simulated unitaries
D = 2

# Hamiltonian Description
J = [0,0,1]
B = 0
hm_list = hamiltonians.short_range_heisenberg(nbits, J, B)
h_name = 'Short Range Heisenberg - {} qubits'.format(nbits)
h_params = 'J=[{:0.2f},{:0.2f},{:0.2f}], B={}'.format(J[0],J[1],J[2],B)

# Initial State of the run
init_sv = Statevector.from_label('01'*(nbits//2))
init_circ = None

# Set the logging paths
param_path = '{}/{}/D={}/N={}/db={}/delta={}/'.format(h_name, h_params,D,N,db,delta)
log_path = './qite_logs/ideal_qite/' + param_path
fig_path = './figs/ideal_qite/' + param_path
run_id = 'run'

params = qite_params()
params.initialize(hm_list, nbits, D)
params.set_run_params(db, delta, N, 0, None, init_sv, init_circ)
params.set_identifiers(log_path,fig_path,run_id)

# Run Flags
time_flag = True # True if you want to log the iteration times

# Plotting Flags
#   Note: calculating the spectrum of the Hamiltonian involves converting it to a matrix
#   of dimension 2**nbits, enable these flags accordingly
eig_flag = True  # True if you want to plot the energy levels of the Hamiltonian
prob_flag = True # True if you want to plot the ground state probability during the run

# Logging Flags


E,times,statevectors,alist = qite(params, time_flag)

plot_data('{}\n{}'.format(h_name,h_params), '001', params, E, statevectors, eig_flag, prob_flag)
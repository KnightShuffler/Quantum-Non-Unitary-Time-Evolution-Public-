import sys
import os
import numpy as np
from random_hamiltonians import *

num_expts = 10

# Hamiltonian Properties
k=2
d=1
l=2
qubit_map=None
Ds=[1,2]
T = 1.0
dts = np.array([0.1,0.05,0.01])
delta = 0.1
num_shots = [0,1,10,100,1000]
init_sv = np.zeros(2**(l**d),dtype=complex)
init_sv[0] = 1.0
objective_meas_list = []
run_id = ''
log_file = 'run.log'

run_params = Run_Params(
    num_expts,
    k,l,d,qubit_map,
    Ds, 
    dts, T,
    num_shots,
    init_sv,
    objective_meas_list=objective_meas_list
)

if not os.path.exists(run_params.log_path):
    os.makedirs(run_params.log_path)

f = open(run_params.log_path + log_file, 'w')
original_stdout = sys.stdout

try:
    sys.stdout = f
    print('Started execution at', time.strftime('%I:%M:%S %p, %d %b %Y', time.localtime()))
    print(run_params.separator)
    start_log(run_params)
    run_random_hamiltonian_experiments(run_params)
    print('Generating data for the figures at', time.strftime('%I:%M:%S %p, %d %b %Y', time.localtime()))
    generate_figure_data(run_params)
    print(run_params.separator)
    print('Generating figures at', time.strftime('%I:%M:%S %p, %d %b %Y', time.localtime()))
    generate_figures(run_params)
    print(run_params.separator)
    print('Finished execution at', time.strftime('%I:%M:%S %p, %d %b %Y', time.localtime()))

except Exception as e:
    print(run_params.separator)
    print('{}: {}'.format(type(e).__name__, e))
    print(traceback.format_exc())
    print('Terminating run at', time.strftime('%I:%M:%S %p, %d %b %Y', time.localtime()))
finally:
    sys.stdout = original_stdout
    f.close()

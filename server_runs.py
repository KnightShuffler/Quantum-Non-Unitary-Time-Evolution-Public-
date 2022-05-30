from ideal_qite import qite, CP_IMPORT_FLAG
from qite_params import *
import hamiltonians
from log_data import log_data, plot_data, plot_all_drifts

from qiskit import Aer
from qiskit.quantum_info import Statevector

num_bits = [2,4,6,8,10]

# Hamiltonian Description
# Short Range Heisenberg Paramaters
J = [1,1,1]
B = 1
h_params = 'J=[{:0.2f},{:0.2f},{:0.2f}], B={}'.format(J[0],J[1],J[2],B)

init_circ = None

# Static Run Parameters
db = 0.1
delta = 0.1

N = 30
D_min = 2
D_max = max(num_bits)

# GPU Usage Flags
gpu_solver_flag = True
gpu_simulator_flag = True

sv_sim = Aer.get_backend('statevector_simulator')

# The number of runs for each drift type (besides none):
run_cap = 1000
padding = int(np.floor(np.log10(run_cap)))

# Plotting Flags
gs_flag = True
prob_flag = False

params = QITE_params()

drift_types = [DRIFT_NONE, DRIFT_A, DRIFT_THETA_2PI, DRIFT_THETA_PI_PI]

def get_drift_string(drift_type):
    drift_string = 'drift_'
    if drift_type == DRIFT_NONE:
        drift_string += 'none'
    elif drift_type == DRIFT_A:
        drift_string += 'a'
    elif drift_type == DRIFT_THETA_2PI:
        drift_string += 'theta_2pi'
    elif drift_type == DRIFT_THETA_PI_PI:
        drift_string += 'theta_pi_pi'
    return drift_string

drift_names = [get_drift_string(d) for d in drift_types]

for nbits in num_bits:
    hm_list = hamiltonians.short_range_heisenberg(nbits, J, B)
    h_name = 'Short Range Heisenberg - {} qubits'.format(nbits)
    param_path = '{}/{}/'.format(h_name, h_params)
    log_path = './qite_logs/ideal_qite/server_runs/'+param_path
    fig_path = './figs/ideal_qite/server_runs/'+param_path

    init_sv = Statevector.from_label('01'*(nbits//2))
    for D in range(D_min, min(D_max,nbits)+2, 2):
        params.load_hamiltonian_params(hm_list, nbits, D)

        for drift_type in drift_types:
            num_runs = 1 if drift_type is DRIFT_NONE else run_cap
            run_name = get_drift_string(drift_type)

            params.set_run_params(db, delta, N, 0, 
            sv_sim, init_circ, init_sv, drift_type, 
            gpu_simulator_flag, gpu_solver_flag & CP_IMPORT_FLAG)
            params.set_identifiers(log_path, fig_path, run_name)

            for run in range(num_runs):
                run_id = str(run).zfill(padding)

                print('Starting QITE run {}-{}, D={}:'.format(run_name, run_id, D))
                E,times,statevectors,alist = qite(params)

                log_data(run_id, params, E, times, alist)
                plot_data('{}\n{}'.format(h_name,h_params), run_id, params, E, statevectors, gs_flag, prob_flag)

        # Plot the data from all the drift types:
        plot_all_drifts(params, '{}\n{}'.format(h_name,h_params), drift_names, run_cap, padding)
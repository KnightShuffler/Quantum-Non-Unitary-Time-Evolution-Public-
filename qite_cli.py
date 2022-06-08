from ideal_qite import CP_IMPORT_FLAG
from ideal_qite import qite as ideal_qite
from qite_params import *
import hamiltonians
from helpers import *
from log_data import log_data, plot_data, plot_all_drifts

from qiskit import Aer
from qiskit.quantum_info import Statevector

import argparse

def positive_int(x: str) -> int:
    try:
        val = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError('Expected integer value received {}'.format(x))
    if val <= 0:
        raise argparse.ArgumentTypeError('Expected positive integer, received {}'.format(val))
    return val

parser = argparse.ArgumentParser(description='Run the QITE implementation.', formatter_class=argparse.RawTextHelpFormatter)
# QITE Run Parameters
run_params = parser.add_argument_group('QITE Run Parameters')
run_params.add_argument('-db', '--timestep', type=float, required=True, 
                    help='Size of imaginary time step')
run_params.add_argument('-delta', '--regularizer', type=float, required=True, 
                    help='Regularizer for solving the linear equations')
run_params.add_argument('-N', '--iterations', type=positive_int, required=True, 
                    help='Number of imagainary time steps to simulate')
run_params.add_argument('-D', '--domain_size', type=positive_int, required=True,
                    help='Maximum domain size for the unitary simulations')
run_params.add_argument('-n', '--nbits', type=positive_int, required=True, 
                    help='Number of qubits in the system')

# Hamiltonian loading
hamiltonian_group = run_params.add_mutually_exclusive_group()
hamiltonian_group.add_argument('-H', '--hamiltonian', type=str,
                                choices=['sr_heisenberg', 'lr_heisenberg', 'afmt_ising'],
                                help='Name of the Hamiltonian Model\n\
\t* sr_heisenberg - Short Range Heisenberg Model\n\
\t* lr_heisenberg - Long Range Heisenberg Model\n\
\t* afmt_ising - Antiferromagnetic Transverse Ising Model')
hamiltonian_group.add_argument('-Hf', '--hamiltonian_file', type=str, 
                                help='File path containing a Hamiltonian description')

run_params.add_argument('-J', '--heisenberg_J', dest='J', metavar='j', nargs=3, type=float, 
                    help='Coupling constants for the Heisenberg and Trasnverse Field Ising Models')
run_params.add_argument('-B', '--heisenberg_B', dest='B', default=0.0, metavar='b', 
                    help='Magnetic field for the Heisenberg and Transverse Field Ising Models')

# Initial Circuit/State
inits = run_params.add_mutually_exclusive_group()
inits.add_argument('--init_sv', type=str, 
                    help='File containing the initial statevector for the QITE run')
inits.add_argument('--init_circ', type=str, 
                    help='File containing the description of the initialization circuit')

# Drift Type and number of runs
run_params.add_argument('--drift', choices=['none', 'a', 'theta_2pi', 'theta_pi_pi'], 
                    help='Specify the drift type', default='none')
run_params.add_argument('--runs', type=positive_int, default=1, 
                    help='Number of times to run the QITE with these parameters')

# GPU usage Flags
parser.add_argument('--gpu_solve', action='store_true',
                    help='Set this to use the GPU to solve the linear equations')
parser.add_argument('--gpu_sim', action='store_true',
                    help='Set this to use the GPU to simulate the quantum circuits')

# Backend Parameters
backend_options = parser.add_argument_group('Quantum Backend Options')
backend_options.add_argument('--backend', choices=['sv_sim, aer_sim'], default='sv_sim', 
                            help="Specify the backend to run the circuits on:\n\
\t* sv_sim - Qiskit's Statevector Simulator\n\
\t* aer_sim - Qiskit's AER Simulator")
backend_options.add_argument('--num_shots', type=positive_int, default=1000, 
                            help='The number shots taken for each measurement in the algorithm')

# Logging Flags
log_flags = parser.add_argument_group('Plotting and Logging Options')
log_flags.add_argument('--plot', action='store_true',
                    help='Set this save a plot of the Energy of the QITE run')
log_flags.add_argument('--plot_gs', action='store_true',
                    help='Set this to calculate and draw the ground state energy in the plot')
log_flags.add_argument('--plot_prob', action='store_true',
                    help='Set this to calculate and plot the fidelity to the ground state')
log_flags.add_argument('--plot_path', type=str, help='Base path to store the plot')
log_flags.add_argument('--log_path', type=str, required=True,
                    help='Base path to store the logged data')
log_flags.add_argument('--run_name', type=str, default='run', help='Identifier of the run logs')

def get_hamiltonian_list(args):
    if args.hamiltonian != None:
        if args.hamiltonian == 'sr_heisenberg':
            if args.J == None:
                raise ValueError('Coupling constants J not specified')
            h_name = 'Short Range Heisenberg - {} qubits'.format(args.nbits)
            hm_list = hamiltonians.short_range_heisenberg(args.nbits, args.heisenberg_J, 
                                                        args.heisenberg_B)
            h_params = 'J = [{:0.2f}, {:0.2f}, {:0.2f}], B = {:0.2f}'.format(args.J[0], args.J[1], args.J[2], args.B)
        elif args.hamiltonian == 'lr_heisenberg':
            if args.J == None:
                raise ValueError('Coupling constants J not specified')
            h_name = 'Long Range Heisenberg - {} qubits'.format(args.nbits)
            hm_list = hamiltonians.long_range_heisenberg(args.nbits, args.heisenberg_J)
            h_params = 'J = [{:0.2f}, {:0.2f}, {:0.2f}]'.format(args.J[0], args.J[1], args.J[2])
        elif args.hamiltonians == 'afmt_ising':
            if args.J == None:
                raise ValueError('Coupling constants J not specified')
            h_name = 'AFM Transverse Field Ising - {} qubits'.format(args.nbits)
            hm_list = hamiltonians.afm_transverse_field_ising(args.nbits, args.heisenberg_J, 
                                                            args.heisenberg_B)
            h_params = 'J = [{:0.2f}, {:0.2f}, {:0.2f}], B = {:0.2f}'.format(args.J[0], args.J[1], args.J[2], args.B)
    else:
        if args.hamiltonian_file != None:
            # TODO: Load hamiltonian from file
            ...
        else:
            raise ValueError('Hamiltonian description not given, make sure to set either the -H \
            or -Hf flags.')
    return hm_list, h_name, h_params

def get_backend(args):
    if args.backend == 'sv_sim':
        backend = Aer.get_backend('statevector_simulator')
    elif args.backend == 'aer_sim':
        backend = Aer.get_backend('aer_simulator')
    
    return backend

def get_inits(args):
    init_circ = None
    init_sv = None

    if args.init_sv != None:
        # TODO: load statevector from file
        ...
    else:
        if args.init_circ != None:
            # TODO: load circuit from file
            ...
        
    return init_circ, init_sv

def get_drift_type(args):
    if args.drift == 'none':
        return DRIFT_NONE
    elif args.drift == 'a':
        return DRIFT_A
    elif args.drift == 'theta_2pi':
        return DRIFT_THETA_2PI
    elif args.drift == 'theta_pi_pi':
        return DRIFT_THETA_PI_PI

def main() -> None:
    args = parser.parse_args()
    # Obtain the Hamiltonian List
    hm_list,h_name,h_params = get_hamiltonian_list(args)    

    # Load the Hamiltonian Params
    params = QITE_params()
    params.load_hamiltonian_params(hm_list, args.nbits, args.D)
    # Load the Run Params
    backend = get_backend(args)
    init_circ, init_sv = get_inits(args)
    drift_type = get_drift_type(args)

    params.set_run_params(args.timestep, args.regularizer, args.iterations, args.num_shots, backend,
    init_circ, init_sv, drift_type, args.gpu_sim, args.gpu_solve and CP_IMPORT_FLAG)
    
    # Set the run/log identifiers
    param_path = '/{}/{}/'.format(h_name,h_params)
    params.set_identifiers(args.log_path+param_path, args.plot_path+param_path, 
                            '{}-{}'.format(args.run_name,args.drift))
    # Run QITE
    padding = int(np.floor(np.log10(args.runs)) + 1)
    for run in range(args.runs):
        run_id = str(run).zfill(padding)
        print('Starting QITE run {} of {}, with drift type: {}'.format(run+1, args.runs, args.drift))
        
        if args.backend == 'sv_sim':
            E,times,statevectors,alist = ideal_qite(params)
        else:
            # TODO: placeholder for QITE code that runs on any backend
            statevectors = None
            ...
        # Plot/Log
        if args.plot:
            plot_data('{}\n{}'.format(h_name,h_params), run_id, params, E, statevectors, 
                        args.plot_gs, args.plot_prob and (args.backend == 'sv_sim'))
        log_data(run_id, params, E, times, alist)

if __name__ == '__main__':
    main()
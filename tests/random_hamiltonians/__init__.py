import h5py
import numpy as np

import logging

from qnute.simulation.parameters import QNUTE_params as Params
from qnute.simulation.output import QNUTE_output as Output
from qnute.simulation.qiskit_sim import qnute
from qnute.simulation.numerical_sim import get_theoretical_evolution
from qnute.helpers import fidelity

from qnute.hamiltonian import Hamiltonian

from .generate_hamiltonians import generate_random_hamiltonians
from .hdf5_io import save_hamiltonians, read_hamiltonians_from_file
from .experiment_params import Experiment_Params
from .utils import calc_qnute_evolution, calc_theoretical_evolution, \
calculate_fidelities, calculate_fidelity_stats, TYPE_stats

def loop_expt_params(EXPT_PARAMS: dict):
    '''Generator that loops through all of the possible Experiment parameter 
    configurations'''
    for k in EXPT_PARAMS['k']:
        for D in EXPT_PARAMS['D']:
            for (i,dt) in enumerate(EXPT_PARAMS['dt']):
                N=int(np.ceil(EXPT_PARAMS['T']/dt))
                for delta in EXPT_PARAMS['delta']:
                    for tta in EXPT_PARAMS['tta']:
                        for tth in EXPT_PARAMS['tth']:
                            for trotter_flag in EXPT_PARAMS['trotter']:
                                for num_shots in EXPT_PARAMS['shots']:
                                    yield Experiment_Params(k,D,dt,N,delta,num_shots,tta,tth,trotter_flag)

def save_random_hamiltonians(filepath:str, topology_id:str, max_k:int,
                             num_hamiltonians:int):
    '''Saves randomly generated Hamiltonians for a given topology'''
    with h5py.File(f'{filepath}/{topology_id}.hdf5', 'r+') as f:
        for k in range(1,max_k+1):
            (hams, ham_dtype) = generate_random_hamiltonians(topology_id, k,
                                                             num_hamiltonians)
            save_hamiltonians(f, hams, ham_dtype, k)
        
def run_experiments(file:h5py.File, expt_params:Experiment_Params, expt_no:int, t_expt_no:int):
    '''Calculates the state vector evolution of the Hamiltonians for
    each experiment in an hdf5 file'''
    #for (k,D,dt,N,delta,num_shots,tta,tth,trotter_flag,expt_no) in loop_expt_params():
    for (i,H) in enumerate(read_hamiltonians_from_file(file, expt_params.k)):
        H.multiply_scalar(1.0j)
        params = Params(H)
        # logging.debug('Multiplying Hamiltonian by i to check')
        
        params.load_hamiltonian_params(expt_params.D)
        params.set_run_params(expt_params.dt, expt_params.delta, 
                                expt_params.N, expt_params.num_shots, 
                                backend=None, taylor_truncate_a=expt_params.tta,
                                taylor_truncate_h=expt_params.tth,
                                trotter_flag=expt_params.trotter_flag)
        calc_qnute_evolution(file, i, expt_no, params)

        # The theoretical Statevectors
        if f'expt_{expt_no}_num' in file['Statevectors'].keys():
            calc_theoretical_evolution(file, i, H, t_expt_no, params)
        
        calculate_fidelities(file, i, expt_no, t_expt_no, params)
        calculate_fidelity_stats(file, expt_no, params)
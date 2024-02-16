import numpy as np
import h5py

import logging
import time
from datetime import datetime

from .experiment_params import Experiment_Params
from .topology import TOPOLOGIES_DICT
from .import loop_expt_params, save_random_hamiltonians, run_experiments
from .hdf5_io import init_ham_file, init_expt_datasets
from .plots import plot_expts, plot_multiple_expts

topology_ID = '31A'
(num_qubits, lattice_dim, qubit_map) = TOPOLOGIES_DICT[topology_ID]

filepath = 'data/random_hamiltonians/'
filename = filepath + topology_ID + '.hdf5'
num_hams = 1000

EXPT_PARAMS = {
    'k': [1,2],               # Values for locality k
    'D': [2],                 # Values for diameter D
    'dt': [0.1],           # Values for time step dt
    'T': 1.0,                   # Values for max time simulated T
    'delta': [0.1],             # Values for delta
    'shots': [0],
    'tth': [-1, 1],
    'tta': [-1],
    'trotter': [False, True]
}
num_expts = 1
for i,l in EXPT_PARAMS.items():
    if isinstance(l, list):
        num_expts *= len(l)

ONLY_PLOT = False
import matplotlib.pyplot as plt
from .plots import save_fig, plot_line

def main():
    logging.getLogger().setLevel(logging.INFO)
    if not ONLY_PLOT:
        print('Running random Hamiltonian test')
        # logging.getLogger().setLevel(logging.DEBUG)
        logging.info('Initializing data file %s', filename)
        init_ham_file(filepath, topology_ID, np.max(EXPT_PARAMS['k']))
        logging.info('Generating and saving random Hamiltonians to %s', filename)
        save_random_hamiltonians(filepath, topology_ID, np.max(EXPT_PARAMS['k']), num_hams)
        logging.info('Initializing experiment datasets in file %s', filename)
        init_expt_datasets(filename, EXPT_PARAMS)
    
    t_expt_no = 1
    with h5py.File(filename, 'r+') as f:
        if not ONLY_PLOT:
            i_dt = EXPT_PARAMS['dt'][0]
            i_delta = EXPT_PARAMS['delta'][0]
            for expt_no, expt_params in enumerate(loop_expt_params(EXPT_PARAMS), start=1):
                logging.info('Experiment Number: %i/%i\n%s - Started at %s', expt_no, num_expts, str(expt_params), datetime.now().__str__())

                t1 = time.monotonic()

                if f'expt_{expt_no}_num' in f['Statevectors'].keys():
                    t_expt_no = expt_no
                stats = f[f'Fidelities/Stats/expt_{expt_no}']
                # TODO: Verify if current expt_params are valid 
                # for the experiment
                run_experiments(f, expt_params, expt_no, t_expt_no)
                if expt_params.dt != i_dt or expt_params.delta != i_delta:
                    # logging.info('Plotting data')
                    nrows = len(EXPT_PARAMS['k'])
                    ncols = len(EXPT_PARAMS['tth'])
                    # fig,axs = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=(ncols*8, nrows*5))
                    # for i,k in enumerate(EXPT_PARAMS['k']):
                    #     for j, tth in enumerate(EXPT_PARAMS['tth']):
                    #         plot_line(axs[i,j], expt_params, stats)
                    
                    flag = nrows>1 and  ncols>1
                    # for (i, k) in enumerate(EXPT_PARAMS['k']):
                    #     (axs[i, 0] if flag else axs[i]).set_ylabel(f'k={k}', rotation=60, size='large', labelpad=15)
                    # for (j, tth) in enumerate(EXPT_PARAMS['tth']):
                    #     (axs[-1, j] if flag else axs[j]).set_xlabel('Full Taylor Series' if tth < 0 
                    #                         else f'{tth}-Term Taylor Series', size='large')
                    
                    # save_fig(i_dt,i_delta,fig,axs[0,0],topology_ID)
                    i_dt = expt_params.dt
                    i_delta = expt_params.delta
                t2 = time.monotonic()
                duration = t2-t1
                units = 'seconds'
                if duration > 60:
                    duration /= 60
                    units = 'minutes'
                if duration > 60:
                    duration /= 60
                    units = 'hours'

                logging.info('Experiment Number: %i/%i finished in %0.2f %s\n', expt_no, num_expts, duration, units)

        logging.info('Creating Plots')
        plot_multiple_expts(f, EXPT_PARAMS)



if __name__ == '__main__':
    main()

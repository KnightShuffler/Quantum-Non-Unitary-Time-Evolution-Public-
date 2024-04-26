import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import logging
import sys

from . import heat_logger
from qnute.simulation.numerical_sim import qnute_logger

from .simulate_1d import (get_zero_bc_analytical_solution,
                          get_zero_bc_frequency_amplitudes,
                          get_periodic_bc_analytical_solution,
                          get_periodic_bc_frequency_amplitudes,
                          run_1D_heat_eqn_simulation)

from .simulate_multidim import (run_heat_eqn_simulation, 
                                get_fourier_amplitudes, 
                                get_analytical_solution)

from .save_experiments import save_experiment_data
from .save_experiments import load_experiment_data
from .plotting import (generate_evolution_and_stats_figure,
                       generate_stats_figure)
from .input_handler import get_inputs

from .rescale_solutions import updateExperimentData


def main():
    heat_logger.setLevel(logging.INFO)
    qnute_logger.setLevel(logging.INFO)

    input_file = sys.argv[1]
    
    delta = 0.1
    filepath = 'data/heat_eqn/alpha=0.8/'
    figpath = 'figs/heat_eqn/alpha=0.8/'

    for expt in get_inputs(input_file):
        heat_logger.info('Running experiment: `%s`', expt.expt_name)
        Nx = 2**expt.num_qbits
        N = np.prod(Nx)
        L = np.zeros(expt.num_space_dims,dtype=np.float64)
        for j in range(expt.num_space_dims):
            L[j] = (Nx[j] + (1 if not expt.periodic_bc_flag[j] else 0))*expt.dx[j]
        min_dx = np.min(expt.dx)
        dt = expt.dtau * min_dx**2
        Nt = np.int32(np.ceil(expt.T/dt))
        times = np.arange(Nt+1)*dt
        
        fourier_amplitudes = get_fourier_amplitudes(expt.f0, expt.num_qbits, expt.periodic_bc_flag)
        analytical_solution = get_analytical_solution(fourier_amplitudes, expt.num_qbits, expt.periodic_bc_flag, expt.dx, L, Nt, dt, expt.alpha)

        qite_solutions = run_heat_eqn_simulation(expt, True)
            
        fidelities = np.zeros((len(expt.D_list), Nt+1), dtype=np.float64)
        log_norm_ratios = np.zeros(fidelities.shape, dtype=np.float64)
        mean_sq_err = np.zeros(fidelities.shape, dtype=np.float64)

        for Di,D in enumerate(expt.D_list):
            for ti,t in enumerate(times):
                fidelities[Di,ti] = np.abs(np.vdot(qite_solutions[Di,ti,:], analytical_solution[ti,:])) / (np.linalg.norm(analytical_solution[ti,:]) * np.linalg.norm(qite_solutions[Di,ti,:]))
                log_norm_ratios[Di,ti] = np.log(np.linalg.norm(analytical_solution[ti,:])) - np.log(np.linalg.norm(qite_solutions[Di,ti,:]))
                mean_sq_err[Di,ti] = np.mean((analytical_solution[ti,:] - qite_solutions[Di,ti,:])**2)

        save_experiment_data(expt.num_qbits, expt.alpha, expt.dx, L, 
                             expt.dtau, Nt, expt.periodic_bc_flag,
                             analytical_solution[0,:], fourier_amplitudes,
                             qite_solutions,analytical_solution,expt.D_list,
                             fidelities,log_norm_ratios,mean_sq_err,
                             filepath=filepath,
                             filename=expt.expt_name,
                             info_string=expt.expt_info)

        expt_data = load_experiment_data(filepath=filepath,filename=expt.expt_name)
        
        if expt.num_space_dims == 1:
            generate_evolution_and_stats_figure(expt_data,
                                                figpath=figpath,
                                                figname=expt.expt_name)
        else:
            generate_stats_figure(expt_data, figpath=figpath, figname=expt.expt_name)
        
        for K in [1,20,50,100]:
            expt_data = updateExperimentData(filepath, expt.expt_name, K)
            figname = expt.expt_name+f'_{K=}'
            if expt.num_space_dims == 1:
                generate_evolution_and_stats_figure(expt_data,
                                                figpath=figpath,
                                                figname=figname)
        else:
            generate_stats_figure(expt_data, figpath=figpath, figname=figname)


if __name__ == '__main__':
    main()
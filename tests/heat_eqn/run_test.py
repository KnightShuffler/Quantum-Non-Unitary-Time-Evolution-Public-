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


def main():
    heat_logger.setLevel(logging.INFO)
    qnute_logger.setLevel(logging.INFO)

    input_file = sys.argv[1]
    
    delta = 0.1
    filepath = 'data/heat_eqn/'
    figpath = 'figs/heat_eqn/'

    for expt in get_inputs(input_file):
        if expt.num_space_dims == 1:
            Nx = 2**expt.num_qbits
            L = (Nx + (1 if not expt.periodic_bc_flag else 0))*expt.dx
            dt = expt.dtau * expt.dx**2
            Nt = np.int32(np.ceil(expt.T/dt))
            times = np.arange(Nt+1)*dt
            sample_x = np.arange(1,Nx+1)*expt.dx if not expt.periodic_bc_flag else np.arange(Nx)*expt.dx
            if not expt.periodic_bc_flag:
                fourier_amplitudes = get_zero_bc_frequency_amplitudes(expt.f0, expt.dx, L)
                analytical_solution = get_zero_bc_analytical_solution(fourier_amplitudes, L, Nx, expt.dx, Nt, dt, expt.alpha)
            else:
                fourier_amplitudes = get_periodic_bc_frequency_amplitudes(expt.f0, expt.dx, L)
                analytical_solution = get_periodic_bc_analytical_solution(fourier_amplitudes, L, Nx, expt.dx, Nt, dt, expt.alpha)
            
            qite_solutions = run_1D_heat_eqn_simulation(expt.num_qbits, expt.dx, expt.T,
                                                        expt.dtau, expt.alpha, expt.f0,
                                                        expt.periodic_bc_flag, expt.D_list, 
                                                        delta, True)
        else:
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


if __name__ == '__main__':
    main()
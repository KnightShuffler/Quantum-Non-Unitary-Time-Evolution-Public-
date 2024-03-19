import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import logging

from . import heat_logger

from .simulate_1d import get_zero_bc_analytical_solution
from .simulate_1d import get_zero_bc_frequency_amplitudes
from .simulate_1d import get_periodic_bc_analytical_solution
from .simulate_1d import get_periodic_bc_frequency_amplitudes
from .simulate_1d import run_1D_heat_eqn_simulation

from .save_experiments import save_experiment_data
from .save_experiments import load_experiment_data
from .plotting import generate_evolution_and_stats_figure


def main():
    heat_logger.setLevel(logging.INFO)
    # n = 3

    reduce_dim_flag = True
    periodic_bc_flag = False
    D_list = list(range(2,n+2,2))

    Nx = 2**n
    dx = 0.1
    L = (Nx + (1 if not periodic_bc_flag else 0))*dx
    T = 1.0
    dtau = 0.1
    dt = dtau*dx*dx
    Nt = np.int32(np.ceil(T/dt))
    alpha = 0.01
    delta = 0.1

    times = np.arange(Nt+1)*dt
    # x = np.arange(Nx+(2 if not periodic_bc_flag else 1))*dx
    x = np.arange(1,Nx+1)*dx if not periodic_bc_flag else np.arange(Nx)*dx

    # psi0 = np.ones(Nx) * 0.5

    frequency_amplitudes = np.zeros(Nx,dtype=np.float64)
    psi0 = np.zeros(Nx,dtype=np.float64)
    for k in range(1,Nx+1,2):
        frequency_amplitudes[k] = 4.0/(k*np.pi)
        psi0 += frequency_amplitudes[k] * np.sin(k*np.pi*x/L)
    analytical_solution = get_zero_bc_analytical_solution(frequency_amplitudes, L,Nx,dx,Nt,dt,alpha)
    

    # if not periodic_bc_flag:
    #     frequency_amplitudes = get_zero_bc_frequency_amplitudes(psi0, dx, L)
    #     analytical_solution = get_zero_bc_analytical_solution(frequency_amplitudes, L, Nx, dx, Nt, dt, alpha)
    # else:
    #     frequency_amplitudes = get_periodic_bc_frequency_amplitudes(psi0, dx, L)
    #     analytical_solution = get_periodic_bc_analytical_solution(frequency_amplitudes, L, Nx, dx, Nt, dt, alpha)
    
    qite_solutions = run_1D_heat_eqn_simulation(n, dx, T, dtau, alpha, psi0, 
                                                periodic_bc_flag, D_list, delta, reduce_dim_flag).real
    
    fidelities = np.zeros((len(D_list), Nt+1), dtype=np.float64)
    log_norm_ratios = np.zeros(fidelities.shape, dtype=np.float64)
    mean_sq_err = np.zeros(fidelities.shape, dtype=np.float64)
    

    for Di,D in enumerate(D_list):
        for ti,t in enumerate(times):
            fidelities[Di,ti] = np.abs(np.vdot(qite_solutions[Di,ti,:], analytical_solution[ti,:])) / (np.linalg.norm(analytical_solution[ti,:]) * np.linalg.norm(qite_solutions[Di,ti,:]))
            log_norm_ratios[Di,ti] = np.log(np.linalg.norm(analytical_solution[ti,:])) - np.log(np.linalg.norm(qite_solutions[Di,ti,:]))
            mean_sq_err[Di,ti] = np.mean((analytical_solution[ti,:] - qite_solutions[Di,ti,:])**2)
        
    print(f'{psi0 = }')
    print(f'{frequency_amplitudes = }')
    print(f'{analytical_solution[0,:] = }')
    print(f'{qite_solutions[:,0,:] = }')

    
    save_experiment_data(n, alpha, dx, L, dtau, Nt, periodic_bc_flag,
                         analytical_solution[0,:], frequency_amplitudes,
                         qite_solutions,analytical_solution,D_list,
                         fidelities,log_norm_ratios,mean_sq_err,
                         filepath='data/heat_eqn/',
                         filename='6qubit_truncated_square_wave',
                         info_string='f(x,0) = 1.0')
    
    expt_data = load_experiment_data(filepath='data/heat_eqn',filename='6qubit_truncated_square_wave')
    
    generate_evolution_and_stats_figure(expt_data,
                                        figpath='figs/heat_eqn/',
                                        figname='6qubit_truncated_square_wave')
    


if __name__ == '__main__':
    main()
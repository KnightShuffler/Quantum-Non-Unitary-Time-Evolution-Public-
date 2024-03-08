import numpy as np
import matplotlib.pyplot as plt

import os
import logging

from . import get_zero_bc_analytical_solution
from . import get_zero_bc_frequency_amplitudes
from . import get_periodic_bc_analytical_solution
from . import get_periodic_bc_frequency_amplitudes
from . import run_1D_heat_eqn_simulation

# @njit(cache=True)
# def graycode_permute_matrix(num_qbits:int):
#     if num_qbits == 1:
#         return np.eye(2,2, dtype=np.complex128)
#     P_n = graycode_permute_matrix(num_qbits-1)
#     X = np.kron(sigma_matrices[1], np.eye(2**(num_qbits-2)))

#     ul =  np.kron( np.array([[1,0],[0,0]],dtype=np.complex128), P_n ) 
#     lr = np.kron( np.array([[0,0],[0,1]], dtype=np.complex128), np.matmul(X,P_n))
#     return ul + lr

def create_simulation_plot(Nx:int, dx:float, Nt:int, dt:float,
                           periodic_bc_flag:bool,
                           analytical_solution:np.ndarray[float], 
                           D_list:list[int], 
                           qite_solutions:np.ndarray[float],
                           data:np.ndarray[float],
                        #    fidelities:np.ndarray[float],
                        #    log_norm_ratio:np.ndarray[float],
                        #    mean_sq_error:np.ndarray[float],
                           fig_name:str, fig_path:str="figs/heat_eqn/"
                           ) -> None:
    figure = plt.figure(figsize=(16,3*(len(D_list)+1)))
    figure.set_facecolor('white')
    figure.suptitle('QITE simulation of the Heat Equation',fontsize=13)
    
    f1,f2 = figure.subfigures(2,1,height_ratios=(1,len(D_list)))
    
    times = np.arange(Nt+1)*dt
    x = np.arange(Nx+(2 if not periodic_bc_flag else 1))*dx
    f = np.zeros(x.shape,dtype=np.float64)

    ax1 = f1.subplots()
    ax1.set_title('Analytical Solution',fontsize=10)

    ax1.set_xlabel('x')
    ax1.set_xlim(x[0]-dx/2,x[-1]+dx/2)
    ax1.set_xticks(x)
    ax1.grid(True)

    col_titles = ['QITE Approximation', 'Fidelity', 'Log of Norm Ratio', 'Mean Squared Error']
    d_subfigs = f2.subfigures(len(D_list),1)
    for row,subfig in enumerate(d_subfigs):
        subfig.suptitle(f'D={D_list[row]}',fontsize=12)
        axs = subfig.subplots(1,4)
        for col, ax in enumerate(axs):
            ax.set_title(col_titles[col])
            if col == 0:
                # if not periodic_bc_flag:
                #     f[1:-1] = qite_solutions[row,ti,:]
                # else:
                #     f[0:-1] = qite_solutions[row,ti,:]
                #     f[-1] = f[0]
                # ax.plot(x,f)
                ax.set_xlabel('x')
                ax.set_xlim(x[0]-dx/2 + x[-1]+dx/2)
                ax.set_xticks(x)
            else:
                ax.plot(times,data[col-1,row,:])
                ax.set_xlabel('Time')

            ax.grid(True)
        subfig.subplots_adjust(wspace=0.3,hspace=0.5, bottom=0.175, top=0.8)
    
    for ti,t in enumerate(times):
        if not periodic_bc_flag:
            f[1:-1] = analytical_solution[ti,:]
        else:
            f[0:-1] = analytical_solution[ti,:]
            f[-1] = f[0]
        l, = ax1.plot(x, f)
        for Di, D in enumerate(D_list):
            if not periodic_bc_flag:
                f[1:-1] = qite_solutions[Di,ti,:]
            else:
                f[0:-1] = qite_solutions[Di,ti,:]
                f[-1] = f[0]
            d_subfigs[Di].axes[0].plot(x,f,color=l.get_color())


    f1.subplots_adjust(left=0.3, right=0.7, top=0.8, bottom=0.175)

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    if fig_path[-1] != '/':
        fig_path += '/'
    figure.savefig(fig_path+fig_name+'.png')
    plt.show()


def main():
    logging.getLogger().setLevel(logging.INFO)
    n = 4

    reduce_dim_flag = True
    periodic_bc_flag = True
    D_list = list(range(2,n+2,2))

    Nx = 2**n
    dx = 0.1
    L = (Nx + (1 if not periodic_bc_flag else 0))*dx
    T = 0.2
    dtau = 0.1
    dt = dtau*dx*dx
    Nt = np.int32(np.ceil(T/dt))
    alpha = 0.01
    delta = 0.1

    times = np.arange(Nt+1)*dt
    x = np.arange(Nx+(2 if not periodic_bc_flag else 1))*dx

    # psi0 = np.abs(np.random.normal(0.0,1.0,Nx))
    # psi0 = np.sin(2*np.pi*x[0:-1]/L)
    psi0 = np.ones(Nx)

    if not periodic_bc_flag:
        frequency_amplitudes = get_zero_bc_frequency_amplitudes(psi0, dx, L)
        analytical_solution = get_zero_bc_analytical_solution(frequency_amplitudes, L, Nx, dx, Nt, dt, alpha)
    else:
        frequency_amplitudes = get_periodic_bc_frequency_amplitudes(psi0, dx, L)
        analytical_solution = get_periodic_bc_analytical_solution(frequency_amplitudes, L, Nx, dx, Nt, dt, alpha)
    
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
    create_simulation_plot(Nx,dx,Nt,dt,periodic_bc_flag,analytical_solution,D_list,qite_solutions,np.array([fidelities,log_norm_ratios,mean_sq_err]), 'test_plot')


if __name__ == '__main__':
    main()
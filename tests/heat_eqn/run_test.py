import numpy as np
from numba import njit
from qnute.hamiltonian import Hamiltonian
from qnute.hamiltonian.laplacian import generateLaplaceHamiltonian1D, generateGrayCodeLaplacian1D
from qnute.simulation.numerical_sim import qnute
from qnute.simulation.numerical_sim import get_theoretical_evolution as get_qnute_th_evolution
from qnute.simulation.parameters import QNUTE_params as Params
from qnute.simulation.output import QNUTE_output as Output
import matplotlib.pyplot as plt
from qnute.helpers.pauli import sigma_matrices

import logging

# @njit(cache=True)
def graycode_permute_matrix(num_qbits:int):
    if num_qbits == 1:
        return np.eye(2,2, dtype=np.complex128)
    P_n = graycode_permute_matrix(num_qbits-1)
    X = np.kron(sigma_matrices[1], np.eye(2**(num_qbits-2)))

    ul =  np.kron( np.array([[1,0],[0,0]],dtype=np.complex128), P_n ) 
    lr = np.kron( np.array([[0,0],[0,1]], dtype=np.complex128), np.matmul(X,P_n))
    return ul + lr

@njit
def get_theoretical_evolution(L:float,Nx:int,dx:float,Nt:int,dt:float,
                              sv_sample_indices:np.ndarray,
                              homogeneous_flag:bool=False,
                              freq:int=1):
    if not homogeneous_flag:
        theoretical_solution = np.zeros((Nt+1,Nx), dtype=np.float64)
        x = np.arange(Nx)*dx + dx
        for i,t in enumerate(np.arange(Nt+1)*dt):
            theoretical_solution[i,:] = np.sin(freq*np.pi*x/L) * np.exp(-np.power(freq*np.pi/L,2)*t)
        return theoretical_solution
    else:
        theoretical_solution = np.ones(((Nt+1),Nx), dtype=np.float64)
        x = np.arange(Nx//2)*dx + dx
        for i,t in enumerate(np.arange(Nt+1)*dt):
            theoretical_solution[i,sv_sample_indices] = np.sin(freq*np.pi*x/L) * np.exp(-np.power(freq*np.pi/L,2)*t)
        return theoretical_solution

def main():
    logging.getLogger().setLevel(logging.INFO)
    n = 4
    qubit_map = {(i,):(i) for i in range(n)}
    assert n-1 > 1
    fig,axs = plt.subplots(n-1,4,figsize=(4*4,(n-1)*4))

    sv_sample_indices = np.arange(2**(n-1))
    sv_extra_indices = np.array([i for i in range(2**n) if i not in sv_sample_indices])
    homogeneous_flag = False
    graycode_flag = False
    full_circle_flag = False
    reduce_dim_flag = False

    Nx = 2**n
    # L = 1.0
    # dx = L / (Nx+1)
    dx = 0.1
    L = dx * (Nx+1)
    if homogeneous_flag:
        L = dx * (Nx//2 + 1)
    T = 1.0

    dtau = 0.1
    Ntau = np.int32(np.ceil(T / dtau))
    dt = dtau*dx*dx
    Nt = np.int32(np.ceil(T/dt))

    print(f'Nt = {Nt}, Ntau = {Ntau}')

    delta = 0.1
    num_shots=0
    backend=None
    trotter_flag = True

    if not graycode_flag:
        H = generateLaplaceHamiltonian1D(n, 0, 1.0, False, homogeneous_flag)
    else:
        H = generateGrayCodeLaplacian1D(n, 1.0, False, False)
    print(H)
    print(H.pterm_list)
    # print(H.hm_indices)
    print(np.real(H.get_matrix()))

    P_n = graycode_permute_matrix(n)
    iP_n = np.linalg.inv(P_n)

    times = np.arange(Nt+1)*dt
    x = np.arange((Nx if not homogeneous_flag else Nx//2)+2)*dx
    f = np.zeros(x.shape,dtype=np.complex128)
    freq = 1
    theoretical_solution = get_theoretical_evolution(L,Nx,dx,Nt,dt,
                                                    sv_sample_indices,
                                                    homogeneous_flag,
                                                    freq)

    psi0 = theoretical_solution[0,:].copy()
    if graycode_flag:
        psi0 = np.dot(P_n, psi0)

    print()
    print(psi0)
    c0 = np.linalg.norm(psi0)
    c_prime = np.sqrt(np.sum(np.power(psi0[sv_sample_indices],2)) + np.power(2,n-1))
    print(f'c0={c0}\nc1={c_prime}')
    psi0 /= c0
    print(psi0)
    print(psi0/psi0[-1])

    params = Params(H, 1, n, qubit_map)

    for Di,D in enumerate(range(2,n+1)):
        if full_circle_flag:
            u_domains = [[j%n for j in range(i,i+D)] for i in range(n)] if D < n else [list(range(n))]
        else:
            u_domains = [list(range(i,i+D)) for i in range(n-D+1)]
        print('u_domains:', u_domains)

        params.load_hamiltonian_params(D, u_domains, reduce_dim_flag, True)
        params.set_run_params(dtau, delta, Nt, num_shots, backend, init_sv=psi0,trotter_flag=trotter_flag)

        out = qnute(params,log_frequency=100,c0=c0)
        # print(len(out.c_list))

        print('Final State:')
        print(out.svs[-1,:])

        # qnute_svs = get_qnute_th_evolution(np.real(H.get_matrix()), psi0, dtau, Ntau)

        fid = np.zeros(times.shape[0], dtype=np.float64)
        mean_sq_err = np.zeros(times.shape[0], dtype=np.float64)

        for i,t in enumerate(times):
            # if graycode_flag:
            f[1:(Nx if not homogeneous_flag else Nx//2)+1] = theoretical_solution[i, np.arange(Nx) if not homogeneous_flag else sv_sample_indices]
            # else:
            #     f[1:(Nx if not homogeneous_flag else Nx//2)+1] = np.matmul(iP_n, theoretical_solution[i, np.arange(Nx) if not homogeneous_flag else sv_sample_indices])
            l, =axs[Di,0].plot(x, f,label=f't={t:0.3f}')
        # taus = np.arange(Ntau+1)*dtau
        # for i,t in enumerate(times):
            if not graycode_flag:
                f[1:(Nx if not homogeneous_flag else Nx//2)+1] = np.real(out.svs[i, np.arange(Nx) if not homogeneous_flag else sv_sample_indices]) * (np.prod(out.c_list[0:i+1]) if not homogeneous_flag else (1.0 / np.mean(out.svs[sv_extra_indices]) ))
            else:
                f[1:(Nx if not homogeneous_flag else Nx//2)+1] = np.dot(iP_n, np.real(out.svs[i, np.arange(Nx) if not homogeneous_flag else sv_sample_indices]) * (np.prod(out.c_list[0:i+1]) if not homogeneous_flag else (1.0 / np.mean(out.svs[sv_extra_indices]) )))
            axs[Di,1].plot(x,
                        f,
                        label=f'tau={t:0.3f}', color=l.get_color())
            # axs[1].plot(x,qnute_svs[i,:], label=f'tau={t:0.3f}')
            if graycode_flag:
                fid[i] = np.abs(np.vdot(theoretical_solution[i,:]/np.linalg.norm(theoretical_solution[i,:]), np.matmul(iP_n, out.svs[i,:])))
                mean_sq_err[i] = np.mean((theoretical_solution[i,:] - (np.dot(iP_n, out.svs[i,:]) * np.prod(out.c_list[0:i+1])))**2)
            else:
                fid[i] = np.abs(np.vdot(theoretical_solution[i,:]/np.linalg.norm(theoretical_solution[i,:]), out.svs[i,:]))
                mean_sq_err[i] = np.mean((theoretical_solution[i,:] - (out.svs[i,:] * np.prod(out.c_list[0:i+1])))**2)

        axs[Di,2].plot(times, fid)
        axs[Di,3].plot(times, mean_sq_err)

        for i in range(4):
            axs[Di,i].grid(True)

        axs[Di,0].set_ylabel(f'D={D}')
        for i in range(2):
            axs[Di,i].set_xlim(-0.01,L + 0.01)
            axs[Di,i].set_xticks(np.arange(0.0,L+0.01,dx))
            if freq == 1:
                axs[Di,i].set_ylim(-0.01, 1.01)
            else:
                axs[Di,i].set_ylim(-1.01, 1.01)

    column_titles = ['Theoretical Evolution',
    'QITE Evolution',
    'Fidelity in unit evolution',
    'Mean squared error']
    for ax,title in zip(axs[0],column_titles):
        ax.set_title(title)
    column_xlabels = ['x', 'x', 'time', 'time']
    for i,label in enumerate(column_xlabels):
        axs[-1,i].set_xlabel(label)
    # axs[2].set_xlim(-0.01, times[-1] + 0.01)
    # axs[2].set_ylim(-0.01, 1.01)
    fig.suptitle(f'QITE Heat equation for {n} qubits\ndtau={dtau} | dx={dx} | {"Gray Code" if graycode_flag else "Simple Encoding"} | {"Linear" if not full_circle_flag else "Circular"} Topology\n{"Full" if not reduce_dim_flag else "Reduced"} Unitary Operator Set | {"Homogeneous" if homogeneous_flag else "Cartesian"} Coordinates | Frequency = {freq}')
    plt.tight_layout()
    plt.savefig(f'figs/heat_eqn/{n}_qubits_dtau={dtau:0.3f}_dx={dx:0.3f}_{"GrayCode" if graycode_flag else "SimpleEncoding"}_{"Linear" if not full_circle_flag else "Circular"}Topology_{"Full" if not reduce_dim_flag else "Reduced"}UnitaryOperatorSet_{"Homogeneous" if homogeneous_flag else "Cartesian"}Coordinates_Frequency={freq}.png')
    # plt.show()
    

    # print(fid)
    # fig2, ax2 = plt.subplots(1,1,figsize=(8,4))
    # ax2.plot(times, fid, color='black')
    # plt.show()
    


if __name__ == '__main__':
    main()
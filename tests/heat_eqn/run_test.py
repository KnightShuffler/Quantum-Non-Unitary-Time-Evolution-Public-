import numpy as np
from numba import njit
from qnute.hamiltonian import Hamiltonian
from qnute.hamiltonian.laplacian import generateLaplaceHamiltonian1D
from qnute.simulation.numerical_sim import qnute
from qnute.simulation.numerical_sim import get_theoretical_evolution as get_qnute_th_evolution
from qnute.simulation.parameters import QNUTE_params as Params
from qnute.simulation.output import QNUTE_output as Output
import matplotlib.pyplot as plt

import logging

@njit
def get_theoretical_evolution(L:float,Nx:int,dx:float,Nt:int,dt:float, 
                              sv_sample_indices:np.ndarray,
                              homogeneous_flag:bool=False):
    if not homogeneous_flag:
        theoretical_solution = np.zeros((Nt+1,Nx), dtype=np.float64)
        x = np.arange(Nx)*dx + dx
        for i,t in enumerate(np.arange(Nt+1)*dt):
            theoretical_solution[i,:] = np.sin(np.pi*x/L) * np.exp(-np.power(np.pi/L,2)*t)
        return theoretical_solution
    else:
        theoretical_solution = np.ones(((Nt+1),Nx), dtype=np.float64)
        x = np.arange(Nx//2)*dx + dx
        for i,t in enumerate(np.arange(Nt+1)*dt):
            theoretical_solution[i,sv_sample_indices] = np.sin(np.pi*x/L) * np.exp(-np.power(np.pi/L,2)*t)
        return theoretical_solution

def main():
    logging.getLogger().setLevel(logging.INFO)
    n = 4
    # qubit_map = {(i,):(i) for i in range(n)}
    qubit_map = {(i,):(i+1)%n for i in range(n)}
    # D = 2
    # u_domains = [[i,i+1] for i in range(n-1)]

    D = 4
    u_domains = [list(range(i,i+4)) for i in range(n-4+1)]


    # qubit_map = {(0,):1, (1,):2, (2,):3, (3,):4, (4,):0}
    # sv_sample_indices = np.array([0,1,2,3])
    sv_sample_indices = np.arange(2**(n-1))
    sv_extra_indices = np.array([i for i in range(2**n) if i not in sv_sample_indices])
    homogeneous_flag = False

    Nx = 2**n
    # L = 1.0
    # dx = L / (Nx+1)
    dx = 0.05
    L = dx * (Nx+1)
    if homogeneous_flag:
        L = dx * (Nx//2 + 1)
    T = 0.01

    dtau = 0.05
    Ntau = np.int32(np.ceil(T / dtau))
    dt = dtau*dx*dx
    Nt = np.int32(np.ceil(T/dt))

    print(f'Nt = {Nt}, Ntau = {Ntau}')

    delta = 0.1
    num_shots=0
    backend=None
    trotter_flag = True

    H = generateLaplaceHamiltonian1D(n, 0, 1.0, False, homogeneous_flag)
    print(H)
    print(H.pterm_list)
    # print(H.hm_indices)
    print(np.real(H.get_matrix()))

    # print(H.get_hm_pterms(0))
    # norm = np.linalg.norm(H.pterm_list['amplitude'])
    # print(f"norm = {norm:0.5f}\n")
    # print(np.real(H.get_matrix())/norm)

    times = np.arange(Nt+1)*dt
    x = np.arange((Nx if not homogeneous_flag else Nx//2)+2)*dx
    f = np.zeros(x.shape,dtype=np.complex128)
    theoretical_solution = get_theoretical_evolution(L,Nx,dx,Nt,dt,
                                                     sv_sample_indices,
                                                     homogeneous_flag)

    psi0 = theoretical_solution[0,:].copy()
    print()
    print(psi0)
    c0 = np.linalg.norm(psi0)
    c_prime = np.sqrt(np.sum(np.power(psi0[sv_sample_indices],2)) + np.power(2,n-1))
    print(f'c0={c0}\nc1={c_prime}')
    psi0 /= c0
    print(psi0)
    print(psi0/psi0[-1])

    params = Params(H, 1, n, qubit_map)
    params.load_hamiltonian_params(D, u_domains, False, True)
    params.set_run_params(dtau, delta, Nt, num_shots, backend, init_sv=psi0,trotter_flag=trotter_flag)

    out = qnute(params,log_frequency=1,c0=c0)
    # print(len(out.c_list))

    print('Final State:')
    print(out.svs[-1,:])

    # qnute_svs = get_qnute_th_evolution(np.real(H.get_matrix()), psi0, dtau, Ntau)

    fig,axs = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,4))

    for i,t in enumerate(times):
        f[1:(Nx if not homogeneous_flag else Nx//2)+1] = theoretical_solution[i, np.arange(Nx) if not homogeneous_flag else sv_sample_indices]
        l, =axs[0].plot(x, f,label=f't={t:0.3f}')
    # taus = np.arange(Ntau+1)*dtau
    # for i,t in enumerate(times):
        f[1:(Nx if not homogeneous_flag else Nx//2)+1] = np.real(out.svs[i, np.arange(Nx) if not homogeneous_flag else sv_sample_indices]) * (np.prod(out.c_list[0:i+1]) if not homogeneous_flag else (1.0 / np.mean(out.svs[sv_extra_indices]) ))
        axs[1].plot(x,
                    f,
                    label=f'tau={t:0.3f}', color=l.get_color())
        # axs[1].plot(x,qnute_svs[i,:], label=f'tau={t:0.3f}')

    axs[0].set_title('Theoretical Evolution')
    axs[1].set_title('QNUTE Evolution')
    for ax in axs:
        ax.grid(True)
    plt.xlim(-0.01,L + 0.01)
    plt.xticks(np.arange(0.0,L+0.01,dx))
    plt.ylim(-0.01, 1.01)
    plt.show()


if __name__ == '__main__':
    main()
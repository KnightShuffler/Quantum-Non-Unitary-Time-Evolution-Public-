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
def get_theoretical_evolution(L:float,Nx:int,dx:float,Nt:int,dt:float):
    theoretical_solution = np.zeros((Nt+1,Nx), dtype=np.float64)
    x = np.arange(Nx)*dx + dx
    for i,t in enumerate(np.arange(Nt+1)*dt):
        theoretical_solution[i,:] = np.sin(np.pi*x/L) * np.exp(-np.power(np.pi/L,2)*t)
    return theoretical_solution

def main():
    logging.getLogger().setLevel(logging.INFO)
    n = 4
    qubit_map = {(i,):i for i in range(n)}

    Nx = 2**n
    # L = 1.0
    # dx = L / (Nx+1)
    dx = 0.1
    L = dx * (Nx+1)
    T = 1.0

    dtau = 0.1
    Ntau = np.int32(np.ceil(T / dtau))
    dt = dtau*dx*dx
    Nt = np.int32(np.ceil(T/dt))

    print(f'Nt = {Nt}, Ntau = {Ntau}')

    D = 2
    delta = 0.1
    num_shots=0
    backend=None
    trotter_flag = True

    H = generateLaplaceHamiltonian1D(n, 1.0, False, False)
    print(H)
    print(H.pterm_list)
    # print(H.hm_indices)
    print(np.real(H.get_matrix()))

    # print(H.get_hm_pterms(0))
    # norm = np.linalg.norm(H.pterm_list['amplitude'])
    # print(f"norm = {norm:0.5f}\n")
    # print(np.real(H.get_matrix())/norm)

    times = np.arange(Nt+1)*dt
    x = np.arange(Nx)*dx + dx
    theoretical_solution = get_theoretical_evolution(L,Nx,dx,Nt,dt)

    psi0 = theoretical_solution[0,:]
    c0 = np.linalg.norm(psi0)
    psi0 /= c0

    params = Params(H, 1, n, qubit_map)
    params.load_hamiltonian_params(D, False, True)
    params.set_run_params(dtau, delta, Nt, num_shots, backend, init_sv=psi0,trotter_flag=trotter_flag)

    out = qnute(params,log_frequency=1,c0=c0)
    print(len(out.c_list))

    # qnute_svs = get_qnute_th_evolution(np.real(H.get_matrix()), psi0, dtau, Ntau)

    fig,axs = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,4))

    for i,t in enumerate(times):
        l, =axs[0].plot(x, theoretical_solution[i,:],label=f't={t:0.3f}')
    taus = np.arange(Ntau+1)*dtau
    for i,t in enumerate(times):
        axs[1].plot(x,np.real(out.svs[i,:])*np.prod(out.c_list[0:i+1]), label=f'tau={t:0.3f}')
        # axs[1].plot(x,qnute_svs[i,:], label=f'tau={t:0.3f}')

    axs[0].set_title('Theoretical Evolution')
    axs[1].set_title('QNUTE Evolution')
    for ax in axs:
        ax.grid(True)
    plt.xlim(-0.01,L + 0.01)
    plt.xticks(np.arange(dx,(Nx+1)*dx,dx))
    plt.ylim(-0.01, 1.01)
    plt.show()


if __name__ == '__main__':
    main()
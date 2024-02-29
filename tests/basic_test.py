import logging
import time

import numpy as np
import matplotlib.pyplot as plt

from qnute.simulation.numerical_sim import qnute, get_theoretical_evolution
from qnute.hamiltonian import Hamiltonian
from qnute.simulation.parameters import QNUTE_params as Params
from qnute.simulation.output import QNUTE_output as Output
from qnute.hamiltonian.laplacian import generateLaplaceHamiltonian1D

def main():
    logging.getLogger().setLevel(logging.INFO)
    
    num_qbits = 4
    # hm_list = [[
    #             [0,1,5,10, 48,49,53,58],
    #             [-1.0,0.5,0.25,0.25, -1.0,0.5,0.25,0.25],
    #             [0, 1,2]
    #         ]]
    
    dt = 0.01
    delta = 0.1
    T = 1.0
    N = int(np.ceil(T/dt))
    num_shots=0
    backend=None
    psi0 = np.zeros(2**num_qbits, dtype=np.complex128)
    # psi0[0] = 1.0
    Nx = 2**num_qbits
    L = 1.0
    h = L/(Nx +1)
    psi0[0:Nx//2] = np.sin(np.pi/L * np.arange(1,Nx//2+1)*h)
    psi0[Nx//2:] = np.ones(Nx//2)
    psi0 /= np.linalg.norm(psi0)

    # ham = Hamiltonian(hm_list, num_qbits)
    ham = generateLaplaceHamiltonian1D(num_qbits, 0, 1.0, False, True)
    print(ham)

    print(np.real(ham.get_matrix()))
    return

    params = Params(ham,1,num_qbits,None)
    logging.info('Testing Hamiltonian Module\n')
    # ham = params.H
    gs = ham.get_gs()[1]
    # ham.multiply_scalar(-1.0j)

    t_svs = get_theoretical_evolution(ham.get_matrix(), psi0, dt, N)

    print(ham)
    # print(ham.hm_indices)
    print(ham.get_matrix())
    # print(params.h_domains)

    logging.info('Testing Parameters module\n')
    D = 2
    params.load_hamiltonian_params(D, False, True)
    params.set_run_params(dt, delta, N, num_shots, backend, init_sv=psi0, trotter_flag=False)

    logging.info('Testing QNUTE simulation.\n')
    out = qnute(params)

    fid = np.zeros(t_svs.shape,dtype=np.float64)
    for i,sv in enumerate(t_svs):
        fid[i] = np.sqrt(np.abs(np.vdot(sv, out.svs[i,:])))

    times = np.arange(0,N+1)*dt
    # plt.plot(times, fid)
    # plt.ylim(0.0,1.05)
    # plt.grid(True)
    # plt.show()
    # return
    print(psi0)
    print('\n\n')
    print(out.svs)
    
    fig,axs = plt.subplots(1,2,sharex=True,sharey=True,figsize=(12,5))
    for i in range(2**num_qbits):
        li, = axs[0].plot(times, np.real(out.svs[:,i]), label=f'|{i}>')
        # axs[0].axhline(y=np.real(gs[i]),linestyle='--',color=li.get_color())
        axs[0].plot(times, np.real(t_svs[:,i]), linestyle='--', color=li.get_color())
        axs[1].plot(times, np.imag(out.svs[:,i]), label=f'|{i}>',color=li.get_color())
        axs[1].plot(times, np.imag(t_svs[:,i]), linestyle='--', color=li.get_color())
        # axs[1].axhline(y=np.imag(gs[i]),linestyle='--',color=li.get_color())
    # l0, = axs[0].plot(times, np.real(out.svs[:,0]),label='|0>')
    # l1, = axs[0].plot(times, np.real(out.svs[:,1  ]),label='|1>')
    axs[0].set_title('Real part of Amplitudes')
    axs[0].grid(True)
    # axs[0].axhline(y=np.sqrt(0.5),linestyle='--',color=l0.get_color())
    # axs[0].axhline(y=-np.sqrt(0.5),linestyle='--',color=l1.get_color())

    # axs[1].plot(times, np.imag(out.svs[:,0]),label='|0>')
    # axs[1].plot(times, np.imag(out.svs[:,1  ]),label='|1>')
    axs[1].set_title('Imaginary part of Amplitudes')
    axs[1].grid(True)
    
    plt.ylim((-1.0,1.0))
    fig.supxlabel('Time')
    fig.supylabel('Amplitude')
    fig.suptitle('QNUTE Evolution of H=-X')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
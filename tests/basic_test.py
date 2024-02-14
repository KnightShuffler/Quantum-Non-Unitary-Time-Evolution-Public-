import logging
import time

import numpy as np
import matplotlib.pyplot as plt

from qnute.simulation.numerical_sim import qnute
from qnute.hamiltonian import Hamiltonian
from qnute.simulation.parameters import QNUTE_params as Params
from qnute.simulation.output import QNUTE_output as Output

def main():
    ham = Hamiltonian(
        [
            [[1],[1.0],[0]]
        ],
        1,1,
        {(0,):0}
    )
    ham.multiply_scalar(-1.0)
    dt = 0.1
    delta = 0.1
    T = 3.0
    N = int(np.ceil(T/dt))
    num_shots=0
    backend=None
    num_qbits = 1
    psi0 = np.zeros(2**num_qbits, dtype=np.complex128)
    psi0[0] = 1.0

    params = Params(ham)
    params.load_hamiltonian_params(2, False, True)
    params.set_run_params(dt, delta, N, num_shots, backend, init_sv=psi0, trotter_flag=True)
    out = qnute(params)

    fig,axs = plt.subplots(1,2,sharex=True,sharey=True,figsize=(12,5))
    times = np.arange(0,N+1)*dt
    l0, = axs[0].plot(times, np.real(out.svs[:,0]),label='|0>')
    l1, = axs[0].plot(times, np.real(out.svs[:,1  ]),label='|1>')
    axs[0].set_title('Real part of Amplitudes')
    axs[0].grid(True)
    axs[0].axhline(y=np.sqrt(0.5),linestyle='--',color=l0.get_color())
    axs[0].axhline(y=-np.sqrt(0.5),linestyle='--',color=l1.get_color())

    axs[1].plot(times, np.imag(out.svs[:,0]),label='|0>')
    axs[1].plot(times, np.imag(out.svs[:,1  ]),label='|1>')
    axs[1].set_title('Imaginary part of Amplitudes')
    axs[1].grid(True)
    
    plt.ylim((-1.0,1.0))
    fig.supxlabel('Time')
    fig.supylabel('Amplitude')
    fig.suptitle('QNUTE Evolution of H=-X')
    plt.show()

if __name__ == '__main__':
    main()

import numpy as np
from itertools import combinations
from qiskit import Aer

import matplotlib.pyplot as plt

from hamiltonians import Hamiltonian
from qnute_params import QNUTE_params
# from sv_qnute import qnute as sv_qnute
from taylor_qnute import qnute as taylor_qnute
from helpers import *

# Qubit Topology Parameters
d = 1
l_max = 1

# QNUTE Parameters
D = 1
reduce_dim = False
dt = 0.1
delta = 0.1
N = 10
taylor_backend = None
# sv_backend = Aer.get_backend('statevector_simulator')


num_experiments = 1

for l in range(1,l_max+1):
    # Generate initial statevector
    dim = 2**l
    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = 1.0

    for i in range(num_experiments):
        # Generate random hamiltonian
        hm_list = []
        for num_indices in range(1,l+1):
            for index_combo in combinations(list(range(l)), num_indices):
                nops = 4**num_indices
                # amps = np.random.uniform(-1.0, 1.0, nops) + 1.0j*np.random.uniform(-1.0, 1.0, nops)
                amps = [0,0,1j,0]
                hm = [ list(range(nops)), amps , list(index_combo) ]
                hm_list.append(hm)
        H = Hamiltonian(hm_list, d, l)
        H_mat = H.get_matrix()
        # H.print()

        # Calculate the numerical statevectors with matrix exponentiation
        num_svs = np.zeros((N+1,dim), dtype=complex)
        num_svs[0] = psi0
        for j in range(1,N+1):
            t = j*dt
            phi = exp_mat_psi(H_mat*t, num_svs[j-1])
            num_svs[j] = phi / np.linalg.norm(phi)
        
        # Calculate the statevectors with QNUTE
        params = QNUTE_params(H)
        params.load_hamiltonian_params(D, reduce_dim=False, load_measurements=False)
        params.set_run_params(dt, delta, N, 0, taylor_backend, init_sv=psi0)
        times, taylor_svs, alist, Slist, blist = taylor_qnute(params, logging=True)

        # Calculate fidelity between states at each time step
        fid = np.zeros(N+1, dtype=float)
        for j in range(N+1):
            fid[j] = fidelity(taylor_svs[j], num_svs[j])

        # Append the calculated fidelity vector to the corresponding file
        print(fid)

        f = plt.figure(figsize=(10,4))
        axs = [f.add_subplot(121), f.add_subplot(122)]
        t = np.arange(0,N+1,1)*dt

        axs[0].plot(t, np.real(num_svs[:,0]), 'r--')
        axs[0].plot(t, np.real(taylor_svs[:,0]), 'ro-')
        axs[0].plot(t, np.real(num_svs[:,1]), 'b--')
        axs[0].plot(t, np.real(taylor_svs[:,1]), 'bo-')

        axs[1].plot(t, np.imag(num_svs[:,0]), 'r--')
        axs[1].plot(t, np.imag(taylor_svs[:,0]), 'ro-')
        axs[1].plot(t, np.imag(num_svs[:,1]), 'b--')
        axs[1].plot(t, np.imag(taylor_svs[:,1]), 'bo-')

        axs[0].set_ylim(-1.5,1.5)
        axs[1].set_ylim(-1.5,1.5)

        axs[0].set_title('Real Part')
        axs[1].set_title('Imaginary Part')

        axs[0].grid()
        axs[1].grid()

        f.suptitle('Amplitudes during evolution', fontsize=16)
        f.tight_layout()
        f.set_facecolor('white')

        plt.show()

        plt.close()
import numpy as np
import logging
import h5py
import os

from qnute.simulation.numerical_sim import qnute, qnute_logger
from qnute.simulation.parameters import QNUTE_params as Params

from . import BlackScholesInfo, BoundaryConditions, Basis, bs_logger
from .option_models import EuropeanCallFormula, EuropeanPutFormula, EuropeanCallState, EuropeanPutState
from .simulation import run_blackScholes_simulation, rescale_qnute_bs_sols

if __name__=='__main__':
    bs_logger.setLevel(logging.INFO)
    qnute_logger.setLevel(logging.INFO)

    bs_data = BlackScholesInfo(r:=0.04,q:=0,sigma:=0.2,
                           basis:=Basis.S,Smin:=0,Smax:=150,
                           BC:=BoundaryConditions.DOUBLE_LINEAR)

    strike = 50
    T = 3
    Nt = 1000
    dt = T/Nt
    rescale_hamiltonian = True
    
    qubit_counts = [7,8]
    
    if not os.path.exists(data_path:='data/black_scholes/'):
        os.makedirs(data_path)
        
    for n in qubit_counts:
        N = 2**n
        for PUT in [True,False]:
            if PUT:
                VT = EuropeanPutState(Smin,Smax,strike,n)
                analytical_sol = np.array([[EuropeanPutFormula(S,tau,strike,r,sigma) for S in np.linspace(Smin,Smax,N)] for tau in np.linspace(0.0,T,Nt+1)])
            else:
                VT = EuropeanCallState(Smin,Smax,strike,n)
                analytical_sol = np.array([[EuropeanCallFormula(S,tau,strike,r,sigma) for S in np.linspace(Smin,Smax,N)] for tau in np.linspace(0.0,T,Nt+1)])
        
            bs_logger.info('Running %d qubit simulation for %s options', n, 'Put' if PUT else 'Call')
            qnute_sols,qnute_norms = run_blackScholes_simulation(VT, bs_data, n, np.arange(2,n+2,2), T, Nt, normalize_hamiltonian=rescale_hamiltonian)
            rescaled_sols, C_psi = rescale_qnute_bs_sols(VT, bs_data, qnute_sols, qnute_norms, T, rescale_frequency=(rescale_freq:=25))

            with h5py.File(data_path+('put' if PUT else 'call')+f'_{n}_qubits.hdf5','w') as file:
                file.attrs['strike'] = strike
                file.attrs['T'] = T
                file.attrs['Nt'] = Nt
                file.attrs['Smin'] = bs_data.Smin
                file.attrs['Smax'] = bs_data.Smax
                file.attrs['r'] = bs_data.r
                file.attrs['q'] = bs_data.q
                file.attrs['sigma'] = bs_data.sigma
                file.attrs['rescale_freq'] = rescale_freq

                file.create_dataset('rescaled_qnute_sols', qnute_sols.shape, np.float64, data=rescaled_sols)
                file.create_dataset('analytical_sols', analytical_sol.shape, np.float64, data=analytical_sol)
    
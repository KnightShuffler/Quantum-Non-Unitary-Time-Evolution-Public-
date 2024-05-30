import numpy as np
import logging
import h5py
import os
from numbers import Number
from functools import partial
import multiprocessing as mp

from qnute.simulation.numerical_sim import qnute, qnute_logger
from qnute.simulation.parameters import QNUTE_params as Params

from . import BlackScholesInfo, BoundaryConditions, Basis, bs_logger
# from .option_models import EuropeanCallFormula, EuropeanPutFormula, EuropeanCallState, EuropeanPutState
from .option_models import *
from .simulation import run_blackScholes_simulation, rescale_qnute_bs_sols

from enum import Enum
class Option(Enum):
    CALL = 0
    PUT = 1
    BULL = 2
    BEAR = 3
    STRADDLE = 4
    STRANGLE = 5
    BUTTERFLY = 6
    CONDOR = 7

def get_analytical_sols(option:Option, bs_data:BlackScholesInfo,
                        num_qbits:int, 
                        T:float,Nt:int,
                        strike:float|np.ndarray[float],
                        *,a:float=0.0
                        ) -> np.ndarray[float]:
    get_sol = partial(getEuropeanSolutions, 
            Smin=bs_data.Smin,
            Smax=bs_data.Smax,
            num_qbits=num_qbits,
            r=bs_data.r,
            sigma=bs_data.sigma)
    match option:
        case Option.CALL:
            if isinstance(strike, Number):
                K = strike
            if isinstance(strike,np.ndarray):
                K = strike[0]
            comb_data = [('c',K,1)]
        case Option.PUT:
            if isinstance(strike, Number):
                K = strike
            if isinstance(strike,np.ndarray):
                K = strike[0]
            comb_data = [('p',K,1)]
        case Option.BULL:
            assert isinstance(strike,np.ndarray)
            Kmin = np.min(strike[0:2])
            Kmax = np.max(strike[0:2])
            comb_data = [('c',Kmin,1), ('c',Kmax,-1)]
        case Option.BEAR:
            assert isinstance(strike,np.ndarray)
            Kmin = np.min(strike[0:2])
            Kmax = np.max(strike[0:2])
            comb_data = [('p',Kmax,1), ('p',Kmin,-1)]
        case Option.STRADDLE:
            if isinstance(strike, Number):
                K = strike
            if isinstance(strike,np.ndarray):
                K = strike[0]
            comb_data = [('c',K,1), ('p',K,1)]
        case Option.STRANGLE:
            assert isinstance(strike,np.ndarray)
            Kmin = np.min(strike[0:2])
            Kmax = np.max(strike[0:2])
            comb_data = [('p',Kmin,1), ('c',Kmax,1)]
        case Option.BUTTERFLY:
            assert a > 0.0, 'Butterfly option needs a > 0'
            if isinstance(strike, Number):
                K = strike
            if isinstance(strike,np.ndarray):
                K = strike[0]
            comb_data = [('c',K-a,1),('c',K,-2),('c',K+a,1)]
        case Option.CONDOR:
            assert a > 0.0, 'Condor option needs a > 0'
            assert isinstance(strike,np.ndarray)
            Kmin = np.min(strike[0:2])
            Kmax = np.max(strike[0:2])
            comb_data = [('c',Kmin,1),('c',Kmin+a,-1),('c',Kmax,-1),('c',Kmax+a,1)]
        case _:
            raise ValueError('Option not defined yet!')
        
    return np.array([ get_sol(tau=tau, combination_data=comb_data) 
                     for tau in np.linspace(0.0,T,Nt+1)])

def run_option_simulation(option:Option, bs_data:BlackScholesInfo, 
                          T:float, Nt:int, 
                          K:float, K1:float, K2:float,
                          data_path:str) -> None:
    if option in [Option.BUTTERFLY, Option.CONDOR]:
        return

    if option in [Option.PUT,Option.CALL,Option.STRADDLE,Option.BUTTERFLY]:
        strike = np.array([K])
    else:
        strike = np.array([K1,K2])
    
    for n in np.arange(2,6+1):
        N = 2**n
        analytical_sols = get_analytical_sols(option, bs_data, n, T, Nt, strike)
        VT = analytical_sols[0,:]
        bs_logger.info('Running %d qubit simulation for %s options', n, option.name)
        qnute_sols,qnute_norms = run_blackScholes_simulation(VT, bs_data, n, np.arange(2,n+2,2), T, Nt, normalize_hamiltonian=True)
        rescaled_sols, C_psi = rescale_qnute_bs_sols(VT, bs_data, qnute_sols, qnute_norms, T, rescale_frequency=(rescale_freq:=25))

        with h5py.File(data_path+option.name+f'_{n}_qubits.hdf5','w') as file:
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
            file.create_dataset('analytical_sols', analytical_sols.shape, np.float64, data=analytical_sols)

if __name__=='__main__':
    bs_logger.setLevel(logging.INFO)
    qnute_logger.setLevel(logging.INFO)

    bs_data = BlackScholesInfo(r:=0.04,q:=0,sigma:=0.2,
                           basis:=Basis.S,Smin:=0,Smax:=150,
                           BC:=BoundaryConditions.DOUBLE_LINEAR)

    # strike = 50
    K1 = 50
    K2 = 100
    K = 75

    T = 3
    Nt = 500
    dt = T/Nt
    rescale_hamiltonian = True
    
    qubit_counts = np.arange(2,6+1)
    
    if not os.path.exists(data_path:='data/black_scholes/'):
        os.makedirs(data_path)

    num_processes = mp.cpu_count()

    run_sim = partial(run_option_simulation, bs_data=bs_data, T=T, Nt=Nt, K=K, K1=K1, K2=K2,data_path=data_path)

    with mp.Pool(processes=num_processes) as pool:
        options = list(Option)[:6]
        pool.map(run_sim, options)
    
    for option in Option:
        run_option_simulation(option, bs_data, T, Nt, K, K1, K2)
        # if option in [Option.BUTTERFLY, Option.CONDOR]:
        #     continue

        # if option in [Option.PUT,Option.CALL,Option.STRADDLE,Option.BUTTERFLY]:
        #     strike = np.array([K])
        # else:
        #     strike = np.array([K1,K2])
        
        # for n in qubit_counts:
        #     N = 2**n
        #     analytical_sols = get_analytical_sols(option, bs_data, n, T, Nt, strike)
        #     VT = analytical_sols[0,:]
        #     bs_logger.info('Running %d qubit simulation for %s options', n, option.name)
        #     qnute_sols,qnute_norms = run_blackScholes_simulation(VT, bs_data, n, np.arange(2,n+2,2), T, Nt, normalize_hamiltonian=rescale_hamiltonian)
        #     rescaled_sols, C_psi = rescale_qnute_bs_sols(VT, bs_data, qnute_sols, qnute_norms, T, rescale_frequency=(rescale_freq:=25))

        #     with h5py.File(data_path+option.name+f'_{n}_qubits.hdf5','w') as file:
        #         file.attrs['strike'] = strike
        #         file.attrs['T'] = T
        #         file.attrs['Nt'] = Nt
        #         file.attrs['Smin'] = bs_data.Smin
        #         file.attrs['Smax'] = bs_data.Smax
        #         file.attrs['r'] = bs_data.r
        #         file.attrs['q'] = bs_data.q
        #         file.attrs['sigma'] = bs_data.sigma
        #         file.attrs['rescale_freq'] = rescale_freq

        #         file.create_dataset('rescaled_qnute_sols', qnute_sols.shape, np.float64, data=rescaled_sols)
        #         file.create_dataset('analytical_sols', analytical_sols.shape, np.float64, data=analytical_sols)
    
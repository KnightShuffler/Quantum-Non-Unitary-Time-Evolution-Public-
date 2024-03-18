import numpy as np
import h5py

import os
import logging

def save_experiment_data(num_qbits:int,
                         alpha:float,
                         dx:float,L:float,
                         dtau:float,
                         Nt:int,
                         periodic_bc_flag:bool,
                         f0:np.ndarray[float],
                         frequency_amplitudes:np.ndarray[float],
                         qite_sols:np.ndarray[float],
                         analytical_sol:np.ndarray[float],
                         D_list:list[int]|np.ndarray[int],
                         fidelity_data:np.ndarray[float],
                         log_norm_ratio_data:np.ndarray[float],
                         mse_data:np.ndarray[float],
                         *,filepath:str='data/',filename:str='experiment',
                         info_string:str=''):
    if filepath[-1] != '/':
        filepath += '/'
    if not os.path.exists(filepath):
        logging.info('Creating directory "%s"', filepath)
        os.makedirs(filepath)

    Nx = 2**num_qbits
    dt = dtau*dx*dx
    
    with h5py.File(filepath+filename+'.hdf5', 'w') as file:
        file.attrs['num_qbits'] = num_qbits
        file.attrs['alpha'] = alpha
        file.attrs['dx'] = dx
        file.attrs['L'] = L
        file.attrs['dtau'] = dtau
        file.attrs['dt'] = dt
        file.attrs['Nt'] = Nt
        file.attrs['periodic_bc_flag'] = periodic_bc_flag
        file.attrs['info'] = info_string

        file.create_dataset('D_list', len(D_list), dtype=np.uint32, data=np.array(D_list,dtype=np.uint32))

        sv = file.create_group('statevectors')
        sv.create_dataset('f0', f0.shape, np.float64, f0)
        sv.create_dataset('frequency_amplitudes', frequency_amplitudes.shape, np.float64, frequency_amplitudes)
        sv.create_dataset('analytical_sol', (Nt,Nx), np.float64, analytical_sol[1:,:])
        sv.create_dataset('qite_sols', (len(D_list),Nt,Nx), np.float64, qite_sols[:,1:,:])

        stats = file.create_group('stats')
        stats.create_dataset('fidelity', fidelity_data.shape, np.float64, fidelity_data)
        stats.create_dataset('log_norm_ratio', log_norm_ratio_data.shape, np.float64, log_norm_ratio_data)
        stats.create_dataset('mean_square_error', mse_data.shape, np.float64, mse_data)
    
    logging.info('Saved data in %s%s.hdf5', filepath,filename)

from dataclasses import dataclass

@dataclass
class ExperimentData:
    dx:float
    L:float 
    dt:float
    T:float 
    periodic_bc_flag:bool 
    f0:np.ndarray[float]
    qite_sols:np.ndarray[float]
    analytical_sol:np.ndarray[float]
    D_list:np.ndarray[int]
    stat_data:np.ndarray[float]

def load_experiment_data(filepath:str,filename:str)->ExperimentData:
    if filepath[-1] != '/':
        filepath += '/'
    
    with h5py.File(filepath+filename+'.hdf5') as file:
        dx:float = file.attrs['dx']
        L:float = file.attrs['L']
        dt:float = file.attrs['dt']
        Nt:float = file.attrs['Nt']
        T = Nt*dt
        periodic_bc_flag:bool = file.attrs['periodic_bc_flag']
        f0:np.ndarray[float] = file['statevectors/f0'][:]
        qite_sols:np.ndarray[float] = file['statevectors/qite_sols'][:]
        analytical_sol:np.ndarray[float] = file['statevectors/analytical_sol'][:]
        D_list:np.ndarray[int] = file['D_list'][:]
        fidelity_data:np.ndarray[float] = file['stats/fidelity'][:]
        log_norm_ratio_data:np.ndarray[float] = file['stats/log_norm_ratio'][:]
        mse_data:np.ndarray[float] = file['stats/mean_square_error'][:]
    
    return ExperimentData(dx, L, dt, T, periodic_bc_flag,
            f0, qite_sols, analytical_sol, D_list,
            np.array([fidelity_data, log_norm_ratio_data, mse_data]))

if __name__ == '__main__':
    print(load_experiment_data('data/', '3qubit_true_square'))
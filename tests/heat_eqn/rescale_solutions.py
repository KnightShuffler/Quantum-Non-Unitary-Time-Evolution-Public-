import numpy as np
import h5py

from dataclasses import replace

from tests.heat_eqn.save_experiments import ExperimentData, load_experiment_data
from tests.heat_eqn.simulate_multidim import get_fourier_eigenstates

def get_cPrimes(expt_data:ExperimentData, Nt:int)->np.ndarray[float]:
    cPrimes = np.zeros((expt_data.D_list.shape[0], Nt), np.float64)
    ex_norms = np.zeros((expt_data.D_list.shape[0],Nt), np.float64)
    
    for Di,D in enumerate(expt_data.D_list):
        for ti,t in enumerate(np.arange(Nt)*expt_data.dt):
            ex_norms[Di][ti] = np.linalg.norm(expt_data.qite_sols[Di][ti])
            if ti == 0:
                cPrimes[Di][ti] = ex_norms[Di][ti] / np.linalg.norm(expt_data.f0)
            else:
                cPrimes[Di][ti] = ex_norms[Di][ti] / ex_norms[Di][ti-1]
    return cPrimes

def get_CPsis(expt_data:ExperimentData, K:int)->np.ndarray[float]:
    Nt = np.int32(np.ceil(expt_data.T/expt_data.dt))
    times = np.arange(1,Nt+1)*expt_data.dt

    if flag_oneD:=(not isinstance(expt_data.periodic_bc_flag,np.ndarray)):
        eigenstates = get_fourier_eigenstates(np.array([expt_data.num_qbits]),np.array([expt_data.periodic_bc_flag]))
    else:
        eigenstates = get_fourier_eigenstates(expt_data.num_qbits,expt_data.periodic_bc_flag)
    gs,gs_norm,gs_freqs = next(eigenstates)
    gs /= np.linalg.norm(gs)

    braket_gs_f0 = np.dot(gs, expt_data.f0)/np.linalg.norm(expt_data.f0)
    
    if flag_oneD:
        lambda0 = (np.pi*gs_freqs[0]/expt_data.L)
    else:
        lambda0 = 0.0
        for fi,freq in enumerate(gs_freqs):
            lambda0 += np.pi*freq/expt_data.L[fi]
    
    lambda0 = expt_data.alpha * (lambda0**2)

    

    CPsis = np.zeros((expt_data.D_list.shape[0], Nt),np.float64)
    cPrimes = get_cPrimes(expt_data,Nt)

    for Di,D in enumerate(expt_data.D_list):
        for ti,t in enumerate(times):
            if ti == 0:
                CPsis[Di][ti] = cPrimes[Di][0]
            elif (ti+1)%K != 0:
                CPsis[Di][ti] = CPsis[Di][ti-1]*cPrimes[Di][ti]
            else:
                CPsis[Di][ti] = braket_gs_f0/np.dot(gs, expt_data.qite_sols[Di][ti])*np.linalg.norm(expt_data.qite_sols[Di][ti])*np.exp(-lambda0*t)

    CPsis *= np.linalg.norm(expt_data.f0)
    return CPsis

def rescale_qite_sols(expt_data:ExperimentData,
                      CPsis:np.ndarray[float]
                      )-> np.ndarray[float]:
    norm_qite_sols = np.abs(expt_data.qite_sols.copy())
    
    Nt = np.int32(np.ceil(expt_data.T/expt_data.dt))
    times = np.arange(1,Nt+1)*expt_data.dt
    
    for Di,D in enumerate(expt_data.D_list):
        for ti,t in enumerate(times):
            norm_qite_sols[Di][ti] *= CPsis[Di][ti]/np.linalg.norm(norm_qite_sols[Di][ti])
    
    return norm_qite_sols

def calc_new_fidelities(expt_data:ExperimentData,new_qite_sols:np.ndarray[float])->np.ndarray[float]:
    fidelities = np.zeros(expt_data.stat_data[0].shape,np.float64)
    
    Nt = np.int32(np.ceil(expt_data.T/expt_data.dt))
    times = np.arange(1,Nt+1)*expt_data.dt

    for Di,D in enumerate(expt_data.D_list):
        for ti,t in enumerate(times):
            fidelities[Di][ti+1] = np.dot(new_qite_sols[Di][ti],expt_data.analytical_sol[ti])/(np.linalg.norm(new_qite_sols[Di][ti])*np.linalg.norm(expt_data.analytical_sol[ti]))

    fidelities[:,0] = np.ones((expt_data.D_list.shape[0]))

    return fidelities

def calc_new_log_norm_ratios(expt_data:ExperimentData,CPsis:np.ndarray[float])->np.ndarray[float]:
    log_norm_ratios = np.zeros(expt_data.stat_data[1].shape,np.float64)

    Nt = np.int32(np.ceil(expt_data.T/expt_data.dt))
    times = np.arange(1,Nt+1)*expt_data.dt

    for Di,D in enumerate(expt_data.D_list):
        for ti,t in enumerate(times):
            log_norm_ratios[Di][ti] = CPsis[Di][ti]/np.linalg.norm(expt_data.analytical_sol[ti])
    
    return np.log(log_norm_ratios)

def calc_new_mse(expt_data:ExperimentData,new_qite_sols:np.ndarray[float])->np.ndarray[float]:
    mse = np.zeros(expt_data.stat_data[2].shape,np.float64)
    Nt = np.int32(np.ceil(expt_data.T/expt_data.dt))
    times = np.arange(1,Nt+1)*expt_data.dt

    for Di,D in enumerate(expt_data.D_list):
        for ti,t in enumerate(times):
            mse[Di][ti] = np.mean((new_qite_sols[Di][ti] - expt_data.analytical_sol[ti])**2)
        
    return mse

def updateExperimentData(filepath:str,filename:str,K:int) ->ExperimentData:
    expt_data = load_experiment_data(filepath,filename)
    
    CPsis = get_CPsis(expt_data, K)
    new_qite_sols = rescale_qite_sols(expt_data, CPsis)
    new_fidelities = calc_new_fidelities(expt_data, new_qite_sols)
    new_log_norm_ratios = calc_new_log_norm_ratios(expt_data,CPsis)
    new_mse = calc_new_mse(expt_data, new_qite_sols)

    with h5py.File(filepath+filename+'.hdf5','r+') as file:
        grp = file.require_group('statevectors/rescaled_qite_sols')
        dset = grp.require_dataset(f'{K=}',new_qite_sols.shape,np.float64,exact=True)
        dset[...] = new_qite_sols
        
        grp = file.require_group(f'stats/rescaled_stats/{K=}')
        dset = grp.require_dataset('fidelity',new_fidelities.shape,np.float64,exact=True)
        dset[...] = new_fidelities
        dset = grp.require_dataset('log_norm_ratio',new_log_norm_ratios.shape,np.float64,exact=True)
        dset[...] = new_log_norm_ratios
        dset = grp.require_dataset('mean_square_error',new_mse.shape,np.float64,exact=True)
        dset[...] = new_mse
        

    new_data = replace(expt_data)
    new_data.qite_sols = new_qite_sols
    new_data.stat_data[1] = new_log_norm_ratios
    new_data.stat_data[2] = new_mse

    return new_data

if __name__ == '__main__':
    from .plotting import generate_evolution_and_stats_figure

    filepath = 'data/heat_eqn/alpha=0.8/'
    filenames = ['6 qubit square wave alpha=0.8', '6 qubit triangle wave alpha=0.8']
    K = 50

    figpath = 'figs/heat_eqn/alpha=0.8/'
    fignames = [filename+f'_{K=}' for filename in filenames]

    for fi,filename in enumerate(filenames):
        expt_data = updateExperimentData(filepath,filename,K)
        generate_evolution_and_stats_figure(expt_data,figpath=figpath,figname=fignames[fi])
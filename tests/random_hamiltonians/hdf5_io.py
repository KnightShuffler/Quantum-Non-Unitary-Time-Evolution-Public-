import h5py
import numpy as np

import logging

from qnute.helpers.lattice import get_k_local_domains_in_map
from qnute.simulation.parameters import QNUTE_params
from qnute.simulation.output import QNUTE_output
from qnute.hamiltonian import Hamiltonian
from .topology import TOPOLOGIES_DICT
from .generate_hamiltonians import generate_random_hamiltonians
from .generate_hamiltonians import ham_dtype_to_Hamiltonian
from .experiment_params import Experiment_Params
from .utils import TYPE_stats

def init_ham_file(filepath:str, topology_id:str, max_k:int):
    logging.debug('Initializing hamiltonian file %s%s.hdf5', filepath, topology_id)
    (num_qubits, lattice_dim, qubit_map) = TOPOLOGIES_DICT[topology_id]
    lattice_bound = np.max(list(qubit_map.values())) + 1

    map_dtype = np.dtype([('index', 'i4'), ('coords', 'i4', (lattice_dim,))])
    map_data = np.zeros(len(qubit_map), dtype=map_dtype)
    for (i,key) in enumerate(qubit_map):
        map_data['index'][i] = key
        map_data['coords'][i] = np.array(qubit_map[key], dtype='i4')

    with h5py.File(f'{filepath}/{topology_id}.hdf5','w') as f:
        map_dset = f.create_dataset('qubit_map', shape=map_data.shape, dtype=map_dtype, data=map_data)
        logging.debug('Creating Hamiltonians group')
        grp = f.create_group('Hamiltonians')
        logging.debug('Setting Hamiltonians group attributes')
        grp.attrs['topology_id'] = topology_id
        grp.attrs['num_qubits'] = num_qubits
        grp.attrs['lattice_dim'] = lattice_dim
        grp.attrs['lattice_bound'] = lattice_bound
        grp.attrs['max_k'] = max_k
        
        logging.debug('Creating Statevectors group')
        f.create_group('Statevectors')
        logging.debug('Creating Fidelities group')
        f.create_group('Fidelities')
        logging.debug('Creating Fidelities/Stats group')
        f.create_group('Fidelities/Stats')

def save_hamiltonians(file:h5py.File, hams:np.array, ham_dtype:np.dtype,
                      k_local:int):
    assert (k_local > 0 and k_local <= file['Hamiltonians'].attrs['max_k'])
    
    if f'k_{k_local}' not in file['Hamiltonians'].keys() :
        dset = file['Hamiltonians'].create_dataset(f'k_{k_local}', shape=hams.shape, dtype=ham_dtype, data=hams)
    else:
        dset = file[f'Hamiltonians/k_{k_local}']
        dset[:] = hams
    
    dset.attrs['k_local'] = k_local
    dset.attrs['num_terms'] = hams.shape[1]
    dset.attrs['num_hamiltonians'] = hams.shape[0]

def init_expt_datasets(filename:str, EXPT_PARAMS: dict):
    logging.debug('Initializing Experirment Statevector and Fidelity Datasets')
    logging.debug('file: %s', filename)
    with h5py.File(filename, 'r+') as file:
        max_k = file['Hamiltonians'].attrs['max_k']
        num_qubits = file['Hamiltonians'].attrs['num_qubits']
        sv_grp = file['Statevectors']
        fid_grp = file['Fidelities']
        expt_no = 1
        for k in EXPT_PARAMS['k']:
            num_hamiltonians = file[f'Hamiltonians/k_{k}'].attrs['num_hamiltonians']
            for D in EXPT_PARAMS['D']:
                for i,dt in enumerate(EXPT_PARAMS['dt']):
                    N = int(np.ceil(EXPT_PARAMS['T'] / dt))
                    shape_ = (num_hamiltonians, N+1, 2**num_qubits)
                    dset = sv_grp.create_dataset(f'expt_{expt_no}_num',
                                            shape=shape_,dtype='c16')
                    dset.attrs['k_local'] = k
                    dset.attrs['D'] = D
                    dset.attrs['dt'] = dt
                    dset.attrs['N'] = N
                    
                    for delta in EXPT_PARAMS['delta']:
                        for num_shots in EXPT_PARAMS['shots']:
                            for tta in EXPT_PARAMS['tta']:
                                for tth in EXPT_PARAMS['tth']:
                                    for trotter_flag in EXPT_PARAMS['trotter']:
                                        dset = sv_grp.create_dataset(f'expt_{expt_no}',
                                                                shape=shape_,
                                                                dtype='c16')
                                        dset.attrs['k_local']=k
                                        dset.attrs['D'] = D
                                        dset.attrs['dt'] = dt
                                        dset.attrs['N'] = N
                                        dset.attrs['delta'] = delta
                                        dset.attrs['shots'] = num_shots
                                        dset.attrs['tta'] = tta
                                        dset.attrs['tth'] = tth
                                        dset.attrs['trotter'] = trotter_flag

                                        dset = fid_grp.create_dataset(f'expt_{expt_no}',
                                                                    shape=(shape_[0],shape_[1]),
                                                                    dtype='f8')
                                        dset.attrs['k_local']=k
                                        dset.attrs['D'] = D
                                        dset.attrs['dt'] = dt
                                        dset.attrs['N'] = N
                                        dset.attrs['delta'] = delta
                                        dset.attrs['shots'] = num_shots
                                        dset.attrs['tta'] = tta
                                        dset.attrs['tth'] = tth
                                        dset.attrs['trotter_flag'] = trotter_flag

                                        dset = fid_grp['Stats'].create_dataset(f'expt_{expt_no}',
                                                                            shape=(shape_[1]),
                                                                            dtype=TYPE_stats)
                                        dset.attrs['k_local']=k
                                        dset.attrs['D'] = D
                                        dset.attrs['dt'] = dt
                                        dset.attrs['N'] = N
                                        dset.attrs['delta'] = delta
                                        dset.attrs['shots'] = num_shots
                                        dset.attrs['tta'] = tta
                                        dset.attrs['tth'] = tth
                                        dset.attrs['trotter_flag'] = trotter_flag
                                        
                                        expt_no += 1

def save_svs(dset:h5py.Dataset, ham_no:int, svs:np.array):
    dset[ham_no][:] = svs


def read_hamiltonians_from_file(file:h5py.File, k_local:int):
    mapdata = file['qubit_map'][:]
    qubit_map = {mapdata['index'][i]:tuple(mapdata['coords'][i]) for i in range(mapdata.shape[0])}

    grp = file['Hamiltonians']
    lattice_bound = grp.attrs['lattice_bound']
    lattice_dim = grp.attrs['lattice_dim']
    num_qubits = grp.attrs['num_qubits']
    topology_id = grp.attrs['topology_id']
    max_k = grp.attrs['max_k']

    dset = grp[f'k_{k_local}']
    num_hamiltonians = dset.attrs['num_hamiltonians']
    num_terms = dset.attrs['num_terms']
    ham_dtype = dset.dtype
    for n in range(num_hamiltonians):
        # ham = dset[n][:]
        yield ham_dtype_to_Hamiltonian(dset[n][:], qubit_map), lattice_dim, lattice_bound

def save_statevectors(filename:str, params:QNUTE_params, out:QNUTE_output):
    with h5py.File(filename, 'a') as f:
        pass

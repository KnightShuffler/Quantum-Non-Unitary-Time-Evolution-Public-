import h5py
import numpy as np

import logging

from qnute.simulation.parameters import QNUTE_params as Params
from qnute.hamiltonian import Hamiltonian

from qnute.simulation.numerical_sim import get_theoretical_evolution
from qnute.simulation.numerical_sim import qnute
# from qnute.simulation.qiskit_sim import qnute

from qnute.helpers import fidelity

TYPE_stats = np.dtype([('mean', 'f8'), ('std', 'f8'), ('q0', 'f8'), 
                       ('q1', 'f8'), ('q2', 'f8'), ('q3', 'f8'), ('q4', 'f8')])


def calc_qnute_evolution(f:h5py.File, ham_no:int, expt_no:int, params:Params):
    out = qnute(params)
    dset = f[f'Statevectors/expt_{expt_no}']
    dset[ham_no] = out.svs

def calc_theoretical_evolution(f:h5py.File, ham_no:int, H:Hamiltonian, expt_no:int, params:Params):
    svs = get_theoretical_evolution(H.get_matrix(), params.init_sv.data, params.dt, params.N)
    dset = f[f'Statevectors/expt_{expt_no}_num']
    dset[ham_no] = svs

def calculate_fidelities(f:h5py.File, ham_no:int, expt_no:int, t_expt_no:int, params:Params):
    logging.debug('ham no: %i expt_no: %i t_expt_no: %i', ham_no, expt_no, t_expt_no)
    dset1 = f[f'Statevectors/expt_{expt_no}'][ham_no]
    dset2 = f[f'Statevectors/expt_{t_expt_no}_num'][ham_no]
    logging.debug('dset1.shape (%i,%i)', dset1.shape[0], dset1.shape[1])
    logging.debug('dset2.shape (%i,%i)', dset2.shape[0], dset2.shape[1])

    dset = f[f'Fidelities/expt_{expt_no}']
    dset[ham_no] = np.array([fidelity(dset1[i], dset2[i]) for i in range(params.N+1)],dtype='f8')

def calculate_fidelity_stats(f:h5py.File, expt_no:int, params:Params):
    dset = f[f'Fidelities/expt_{expt_no}']
    stats = np.zeros(params.N+1, dtype=TYPE_stats)
    stats['mean'] = np.mean(dset,axis=0)
    stats['std'] = np.std(dset,axis=0)
    stats['q0'] = np.min(dset,axis=0)
    stats['q1'] = np.percentile(dset,25,axis=0)
    stats['q2'] = np.percentile(dset,50,axis=0)
    stats['q3'] = np.percentile(dset,75,axis=0)
    stats['q4'] = np.max(dset,axis=0)
    stat_dset = f[f'Fidelities/Stats/expt_{expt_no}']
    stat_dset[:] = stats

import numpy as np

import json
from dataclasses import dataclass
from typing import Any, Generator

from . import heat_logger

@dataclass
class ExperimentInput:
    ndims:int
    alpha:float
    dtau:float
    T:float
    D_list:np.ndarray
    expt_name:str
    expt_info:str
    num_qbits:int|np.ndarray[int]
    dx:float|np.ndarray[float]
    periodic_bc_flag:bool|np.ndarray[bool]
    Nx:int|np.ndarray[int]
    L:float|np.ndarray[float]
    Nt:int
    f0:np.ndarray[float]

def get_inputs(file:str) -> Generator[ExperimentInput]:
    with open(file) as f:
        data = json.load(f)
    if data is not None:
        if isinstance(data, list):
            for entry in data:
                yield parse_entry(entry)
        else:
            yield parse_entry(data)

def parse_entry(entry:dict[str,Any]) -> ExperimentInput:
    ndims = entry['num_space_dims']
    alpha = entry['alpha']
    dtau = entry['dtau']
    T = entry['T']
    D_list = np.array(entry['D_list'],dtype=np.uint32)
    expt_name = entry['expt_name']
    expt_info = entry['expt_info']

    num_qbits = parse_field(ndims, entry, 'num_qbits')
    dx = parse_field(ndims, entry, 'dx')
    periodic_bc_flag = parse_field(ndims, entry, 'periodic_bc_flag')

    if ndims == 1:
        Nx = 2**num_qbits
        L = Nx*dx + (dx if not periodic_bc_flag else 0.0)
        dt = dtau * dx * dx
    else:
        Nx = np.power(2, num_qbits, dtype=np.uint32)
        L = np.zeros(ndims,dtype=np.float64)
        for i in range(ndims):
            L[i] = Nx[i]*dx[i] + (dx[i] if not periodic_bc_flag[i] else 0.0)
        dt = dtau * np.min(dx)**2
    Nt = np.uint32(np.ceil(T/dt))

    f0 = np.load(entry['f0_file'])

    return ExperimentInput(ndims, alpha, dtau, T, D_list, expt_name, expt_info,
                           num_qbits, dx, periodic_bc_flag, Nx, L, Nt, f0)
    

def parse_field(ndims:int, entry:dict, field:str)->np.ndarray|Any:
    if isinstance(entry[field], list):
        if (nfield:=len(entry[field])) < ndims:
            heat_logger.error('In entry name `%s`: Expected %d elements for field `%s`, recieved %d', entry['expt_name'], ndims, field, nfield)
            raise ValueError()
        return np.array(entry['field'])
    else:
        if ndims > 1:
            heat_logger.warning('In entry name `%s`: Only one %s provided, using for all spacial dimensions', entry['expt_name'], field)
            return np.array([entry[field]]*ndims)
        else:
            return entry[field]

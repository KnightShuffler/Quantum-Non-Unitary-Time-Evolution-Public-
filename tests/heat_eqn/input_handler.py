import numpy as np

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Generator, Iterable

from . import heat_logger

@dataclass
class ExperimentInput:
    num_space_dims:int
    alpha:float
    dtau:float
    T:float
    D_list:np.ndarray
    expt_name:str
    expt_info:str
    num_qbits:int|np.ndarray[int]
    dx:float|np.ndarray[float]
    periodic_bc_flag:bool|np.ndarray[bool]
    L:float|np.ndarray[float]
    Nt:int
    f0:np.ndarray[float]

    def dict(self):
        return {k:v if not isinstance(v, np.ndarray) else list(v) for k,v in asdict(self).items()}

def generate_input_file(inputs:Iterable[ExperimentInput], filepath:str,file:str,
                        datapath:str='tests/heat_eqn/'):
    if filepath[-1] != '/':
        filepath += '/'
    if datapath[-1] != '/':
        datapath += '/'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if not os.path.exists(datapath):
        os.makedirs(datapath)

    with open(filepath+file+'.json', 'w') as outfile:
        outfile.write('[\n')
        for i,entry in enumerate(inputs):
            np.save(datafile:=datapath+entry.expt_name+'.npy', entry.f0)
            d = entry.dict()
            d['f0'] = datafile
            json_object = json.dumps(d, indent=4)
            outfile.write(json_object)
            if i != len(inputs) - 1:
                outfile.write(',\n')
        outfile.write('\n]')
    

def get_inputs(file:str) -> Generator[ExperimentInput,None,None]:
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
    D_list = np.array(entry['D_list'],dtype=np.int32)
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
        Nx = np.power(2, num_qbits, dtype=np.int32)
        L = np.zeros(ndims,dtype=np.float64)
        for i in range(ndims):
            L[i] = Nx[i]*dx[i] + (dx[i] if not periodic_bc_flag[i] else 0.0)
        dt = dtau * np.min(dx)**2
    Nt = np.int32(np.ceil(T/dt))

    f0 = np.load(entry['f0'])

    return ExperimentInput(ndims, alpha, dtau, T, D_list, expt_name, expt_info,
                           num_qbits, dx, periodic_bc_flag, L, Nt, f0)
    

def parse_field(ndims:int, entry:dict, field:str)->np.ndarray|Any:
    if isinstance(entry[field], list):
        if (nfield:=len(entry[field])) < ndims:
            heat_logger.error('In entry name `%s`: Expected %d elements for field `%s`, recieved %d', entry['expt_name'], ndims, field, nfield)
            raise ValueError()
        return np.array(entry[field])
    else:
        if ndims > 1:
            heat_logger.warning('In entry name `%s`: Only one %s provided, using for all spacial dimensions', entry['expt_name'], field)
            return np.array([entry[field]]*ndims)
        else:
            return entry[field]

if __name__ == '__main__':
    l = [
        ExperimentInput(1,0.1, 0.1, 1.0, [2,4],
                         'test', 'testinfo', 2, 0.1,
                         False, 0.9, 1000, 
                         np.array([0.0, 1.0, 0.0, 0.0])),
        ExperimentInput(1,0.1, 0.1, 1.0, [2,4],
                         'test2', 'test2info', 2, 0.1,
                         False, 0.9, 1000, 
                         np.array([0.0, 0.0, 1.0, 0.0])),
        ExperimentInput(2,0.1, 0.1, 1.0, [2,4],
                         'test2', 'test2info', [1,1], [0.1,0.2],
                         [False,True], 0.9, 1000, 
                         np.array([0.0, 0.0, 1.0, 0.0]))
        ]
    generate_input_file(l, './', 'test_input')

    for entry in get_inputs('./test_input.json'):
        print(entry,'\n')
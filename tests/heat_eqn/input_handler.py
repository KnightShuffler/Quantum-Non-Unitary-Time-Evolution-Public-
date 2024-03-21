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
    # Nt:int
    f0:np.ndarray[float]

    def dict(self):
        return {k:v if (k=='f0' or not isinstance(v, np.ndarray)) else list(v) for k,v in asdict(self).items()}

def generate_input_file(inputs:Iterable[ExperimentInput]|ExperimentInput, filepath:str,file:str,
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
        if isinstance(inputs, Iterable):
            for i,entry in enumerate(inputs):
                np.save(datafile:=datapath+entry.expt_name+'.npy', entry.f0)
                d = entry.dict()
                d['f0'] = datafile
                json_object = json.dumps(d, indent=4)
                outfile.write(json_object)
                if i != len(inputs) - 1:
                    outfile.write(',\n')
        else:
            np.save(datafile:=datapath+inputs.expt_name+'.npy', inputs.f0)
            d = inputs.dict()
            d['f0'] = datafile
            json_object = json.dumps(d, indent=4)
            outfile.write(json_object)
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

    # if ndims == 1:
    #     dt = dtau * dx * dx
    # else:
    #     dt = dtau * np.min(dx)**2
    # Nt = np.int32(np.ceil(T/dt))

    f0 = np.load(entry['f0'])

    return ExperimentInput(ndims, alpha, dtau, T, D_list, expt_name, expt_info,
                           num_qbits, dx, periodic_bc_flag, f0)
    

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
    a = 1.0
    b = 2.0
    dx = 0.1
    Nx = 8
    L = 8*dx
    triangle_sv = np.array([1.0,1.25,1.5,1.75,2.0,1.75,1.5,1.25])
    
    expts = ExperimentInput(1, 0.05, 0.1, 1.0, [2,4],
                            'Triangle Wave', 'a=1,b=2', 3, 0.1, True, triangle_sv)
    generate_input_file(expts, './', 'test_input')

    for entry in get_inputs('./test_input.json'):
        print(entry,'\n')
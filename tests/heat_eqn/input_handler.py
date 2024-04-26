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
    D_list:list
    expt_name:str
    expt_info:str
    num_qbits:int|list[int]
    dx:float|list[float]
    periodic_bc_flag:bool|list[bool]
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
        return np.array(entry[field][:ndims])
    else:
        if ndims > 1:
            heat_logger.warning('In entry name `%s`: Only one %s provided, using for all spacial dimensions', entry['expt_name'], field)
        return np.array([entry[field]]*ndims)

if __name__ == '__main__':
    expts:list[ExperimentInput] = []

    alpha = 0.01
    dtau = 0.1
    T = 1.0
    D_list = [2,4,6]
    num_qbits = [5,5]
    dx = [0.1,0.1]

    # Square Wave 2d
    f0 = np.ones(2**np.sum(num_qbits), dtype=np.float64)
    expts.append(ExperimentInput(2, alpha, dtau, T, D_list, '2D Square Wave 5x5 qubits',
                                 'f(x,y)=1.0 with zero boundary conditions',
                                 num_qbits, dx, [False,False], f0))
    
    # 2D Triangle Wave
    a = 1.0
    b = 2.0
    h = b-a
    Nx = 2**num_qbits[0]
    slope = 2*(b-a)/Nx
    f1 = np.zeros(Nx,dtype=np.float64)
    f1[0:Nx//2] = slope*np.arange(Nx//2)
    f1[Nx//2:Nx] = h - slope*(np.arange(Nx//2,Nx)-(Nx//2))
    f11 = np.kron(f1,f1) + a

    expts.append(ExperimentInput(2, alpha, dtau, T, D_list, '2D Triangle Wave 5x5 qubits',
                                 'a=1.0, b=2.0, Periodic boundary conditions on each dimension',
                                 num_qbits, dx, [True,True], f11))
    
    # Mixed Boundary conditions
    c = 1.5
    x = np.arange(1, Nx+1)
    g1 = 4.0*c/(Nx+1)**2 * ((Nx+1)*x - x**2) # Inverted Parabola with zeros at x=0 and x=Nx, with max height c at x=Nx/2

    expts.append(ExperimentInput(2, alpha, dtau, T, D_list, 'Inverted Parabola on x, Triangle Wave on y',
                                 'a=1.0, c=1.5',
                                 num_qbits, dx, [False,True], np.kron(f1 + a, g1)))

    # for expt in expts:
    #     print(expt.dict())
    #     print()



    generate_input_file(expts, './', '2d_inputs')

    for entry in get_inputs('./2d_inputs.json'):
        print(entry,'\n')
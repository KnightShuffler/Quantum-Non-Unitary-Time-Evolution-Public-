import numpy as np
import pandas as pd
from qnute_params import QNUTE_params

class QNUTE_output:
    def __init__(self, params: QNUTE_params):
        self.times = np.zeros(params.N)
        self.a_list = []
        # self.S_list = []
        # self.b_list = []
        self.c_list = []
        if params.store_state_vector:
            self.svs = np.zeros((params.N+1, 2**params.nbits),dtype=complex)
        else:
            self.svs = None
        
        self.measurements = {}
        for m in params.objective_measurements:
            self.measurements[m[0]] = np.zeros(params.N+1,dtype=float)
    
    def log_output(self, run_id:str, path:str='./logs/', 
    a_list_flag:bool=False, c_list_flag:bool=True, 
    sv_flag:bool=True, meas_flag:bool=True):
        if path[-1] != '/':
            path += '/'

        if a_list_flag:
            df = pd.DataFrame(self.a_list)
            df.rename(columns={0:'a', 1:'Qubit Domain', 2:'Reduced Dimension'})
            df.to_csv(path+run_id+'_a_list.csv', index=False)
        if c_list_flag:
            c = pd.DataFrame(self.c_list)
            c.to_csv(path+run_id+'_norms.csv', header=False, index=False)
        if sv_flag:
            r_psis = pd.DataFrame(np.real(self.svs))
            r_psis.to_csv(path+run_id+'_statevectors_real.csv',header=False,index=False)
            i_psis = pd.DataFrame(np.imag(self.svs))
            i_psis.to_csv(path+run_id+'_statevectors_imag.csv',header=False,index=False)
        if meas_flag and len(self.measurements) > 0:
            meas = pd.DataFrame(self.measurements)
            meas.to_csv(path+run_id+'_measurements.csv',index=False)

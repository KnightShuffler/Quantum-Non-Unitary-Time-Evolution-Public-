import numpy as np
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
    
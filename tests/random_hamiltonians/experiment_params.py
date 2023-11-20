from dataclasses import dataclass

@dataclass
class Experiment_Params:
    k:int
    D:int
    dt:float
    N:int
    delta:float
    num_shots:int
    tta:int
    tth:int
    trotter_flag:bool

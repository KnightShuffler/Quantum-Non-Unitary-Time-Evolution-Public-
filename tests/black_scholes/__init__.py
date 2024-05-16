import logging
from enum import Enum
from dataclasses import dataclass

bs_logger = logging.getLogger('BlackScholes')

class Basis(Enum):
    S = 0
    X = 1

class BoundaryConditions(Enum):
    ZERO_AFTER = 0
    DIRICHLET_NODE = 1
    LINEAR = 2
    PDE = 3
    DOUBLE_DIRICHLET_NODE = 4
    DOUBLE_LINEAR = 5

@dataclass
class BlackScholesInfo:
    r:float
    q:float
    sigma:float
    basis:Basis
    Smin:float
    Smax:float
    BC:BoundaryConditions

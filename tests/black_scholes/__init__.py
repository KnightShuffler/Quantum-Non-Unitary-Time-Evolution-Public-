import logging
from enum import Enum
from dataclasses import dataclass

bs_logger = logging.getLogger('BlackScholes')

class Basis(Enum):
    S = 0
    X = 1

class BoundaryConditions(Enum):
    ZERO_AFTER = 0
    ZERO_AT = 1
    LINEAR = 2
    PDE = 3

@dataclass
class BlackScholesInfo:
    r:float
    q:float
    sigma:float
    basis:Basis
    Smax:float
    BC:BoundaryConditions

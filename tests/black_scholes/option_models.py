import numpy as np
from scipy.stats import norm

def EuropeanPutState(Smin:float, Smax:float, strike:float, num_qbits:int) -> np.ndarray[float]:
    N = 2**num_qbits
    V = np.where((S:=np.linspace(Smin,Smax,N)) < strike, strike - S, 0.0)
    return V

def EuropeanCallState(Smin:float, Smax:float, strike:float, num_qbits:int) -> np.ndarray[float]:
    N = 2**num_qbits
    V = np.where((S:=np.linspace(Smin,Smax,N)) > strike, S-strike, 0.0)
    return V

def EuropeanCallFormula(S:float, tau:float, strike:float, r:float, sigma:float) -> float:
    if tau == 0.0:
        return np.max([S-strike,0.0])
    d_plus = (np.log(S/strike) + (r+sigma**2/2)*tau) / (sigma*np.sqrt(tau))
    d_minus = d_plus - (sigma*np.sqrt(tau))
    return norm.cdf(d_plus)*S - norm.cdf(d_minus) * strike * np.exp(-r*(tau))

def EuropeanPutFormula(S:float, tau:float, strike:float, r:float, sigma:float):
    if tau == 0.0:
        return np.max([strike-S,0.0])
    return strike*np.exp(-r*tau) - S + EuropeanCallFormula(S,tau,strike,r,sigma)

def BullSpreadPayoff(Smin:float,Smax:float,
                     K1:float,K2:float,
                     num_qbits:int) -> np.ndarray[float]:
    # assert K2 > K1
    Kmin = min(K1,K2)
    Kmax = max(K1,K2)
    return EuropeanCallState(Smin,Smax,Kmin,num_qbits) - EuropeanCallState(Smin,Smax,Kmax,num_qbits)

def BearSpreadPayoff(Smin:float,Smax:float,
                     K1:float,K2:float,
                     num_qbits:int) -> np.ndarray[float]:
    # assert K2 > K1
    Kmin = min(K1,K2)
    Kmax = max(K1,K2)
    return EuropeanPutState(Smin,Smax,Kmax,num_qbits) - EuropeanPutState(Smin,Smax,Kmin,num_qbits)

def StraddlePayoff(Smin:float,Smax:float,
                   K:float,num_qbits:int) ->np.ndarray[float]:
    return EuropeanCallState(Smin,Smax,K,num_qbits) + EuropeanPutState(Smin,Smax,K,num_qbits)

def StranglePayoff(Smin:float,Smax:float,
                   K1:float,K2:float,
                   num_qbits:int) -> np.ndarray[float]:
    Kmin = min(K1,K2)
    Kmax = max(K1,K2)

    return EuropeanPutState(Smin,Smax,Kmin,num_qbits) + EuropeanCallState(Smin,Smax,Kmax,num_qbits)

def ButterflyPayoff(Smin:float,Smax:float,
                    K:float,a:float,
                    num_qbits:int) -> np.ndarray[float]:
    return np.sum(
        [
        EuropeanCallState(Smin,Smax,K-a,num_qbits),
        -2*EuropeanCallState(Smin,Smax,K,num_qbits),
        EuropeanCallState(Smin,Smax,K+a,num_qbits),
        ], axis=0)

def CondorPayoff(Smin:float,Smax:float,
                 K1:float,K2:float,a:float,
                 num_qbits:int) -> np.ndarray[float]:
    Kmin = min(K1,K2)
    Kmax = max(K1,K2)
    return np.sum(
        [
            EuropeanCallState(Smin,Smax,Kmin,num_qbits),
            -EuropeanCallState(Smin,Smax,Kmin+a,num_qbits),
            -EuropeanCallState(Smin,Smax,Kmax,num_qbits),
            EuropeanCallState(Smin,Smax,Kmax+a,num_qbits),
        ],axis=0)

def getEuropeanSolutions(Smin:float,Smax:float,num_qbits:float,
                         r:float,sigma:float,tau:float, 
                        combination_data:list[tuple[str,float,float]])->np.ndarray[float]:
    x = np.linspace(Smin,Smax,2**num_qbits)
    sol = np.zeros(2**num_qbits,np.float64)
    for (type,strike,scale) in combination_data:
        if type.capitalize() == 'C':
            for i,S in enumerate(x):
                sol[i] += scale*EuropeanCallFormula(S, tau, strike, r, sigma)
        else:
            for i,S in enumerate(x):
                sol[i] += scale*EuropeanPutFormula(S, tau, strike, r, sigma)
    return sol
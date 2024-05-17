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

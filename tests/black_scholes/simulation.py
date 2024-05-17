import numpy as np

from . import BlackScholesInfo
from .hamiltonian import generateBlackScholesHamiltonian

from qnute.simulation.numerical_sim import qnute
from qnute.simulation.parameters import QNUTE_params as Params

def run_blackScholes_simulation(maturity_state:np.ndarray[float],
                                bs_data:BlackScholesInfo,
                                num_qbits:int,
                                D_list:np.ndarray[int],
                                T:float,
                                Nt:int,
                                ) -> tuple[np.ndarray[float],np.ndarray[float]]:
    N = 2**num_qbits
    assert maturity_state.shape[0] == N
    dt = T/Nt
    c0 = np.linalg.norm(maturity_state)
    psi0 = maturity_state/c0

    qnute_sols = np.zeros((D_list.shape[0],Nt+1,N), np.float64)
    qnute_norms = np.zeros((D_list.shape[0],Nt+1), np.float64)

    BSHam = generateBlackScholesHamiltonian(bs_data, num_qbits)

    params = Params(BSHam*(-1), 1, num_qbits)
    params.set_run_params(dt,0.1,Nt,0,None,init_sv=psi0,trotter_flag=True)

    for Di,D in enumerate(D_list):
        u_domains = [list(range(i,i+D)) for i in range(num_qbits-D+1)]
        params.load_hamiltonian_params(D, u_domains, True, True)
        out = qnute(params, log_frequency=500, c0=c0)
        qnute_sols[Di,:,:] = np.abs(out.svs.real)
        qnute_norms[Di,:] = np.array(out.c_list)
    return (qnute_sols,qnute_norms)

def rescale_qnute_bs_sols(f0:np.ndarray[float], bs_data:BlackScholesInfo,
                          qnute_sols:np.ndarray[float],qnute_norms:np.ndarray[float],
                          T,*,rescale_frequency:int=1
                          )->tuple[np.ndarray[float],np.ndarray[float]]:
    if f0[0] == 0.0 and f0[-1] == 0.0:
        raise ValueError('Initial state must be non-zero on at least one boundary to rescale with linear boundary conditions')
    
    N = f0.shape[0]
    dS = (bs_data.Smax - bs_data.Smin)/(N-1)
    ND = qnute_sols.shape[0]
    Nt = qnute_sols.shape[1] - 1

    if left_flag:=(f0[0] != 0.0):
        a0l = (f0[1] - f0[0])/dS
        b0l = f0[0] - bs_data.Smin*a0l
        
        if bs_data.Smin > 0.0:
            al = a0l * np.exp(-(bs_data.q/bs_data.Smin)*np.linspace(0,T,Nt+1))
        else:
            al = np.zeros(Nt+1)
        bl = b0l * np.exp(-bs_data.r*np.linspace(0,T,Nt+1))

        C_star_l = np.zeros((ND,int(np.ceil(Nt/rescale_frequency))), np.float64)
        for Di in range(qnute_sols.shape[0]):
            for index,ti in enumerate(np.arange(1,Nt+1,rescale_frequency)):
                C_star_l[Di,index] = (bs_data.Smin*al[ti] + bl[ti])/(qnute_norms[Di,0]*qnute_sols[Di,ti,0])

    if right_flag:=(f0[-1] != 0.0):
        a0r = (f0[-1] - f0[-2])/dS
        b0r = f0[-1] - bs_data.Smax*a0r
        ar = a0r * np.exp(-(bs_data.q/bs_data.Smax)*np.linspace(0,T,Nt+1))
        br = b0r * np.exp(-bs_data.r*np.linspace(0,T,Nt+1))

        C_star_r = np.zeros((ND,int(np.ceil(Nt/rescale_frequency))), np.float64)
        for Di in range(qnute_sols.shape[0]):
            for index,ti in enumerate(np.arange(1,Nt+1,rescale_frequency)):
                C_star_r[Di,index] = (bs_data.Smax*ar[ti] + br[ti])/(qnute_norms[Di,0]*qnute_sols[Di,ti,-1])
    
    if left_flag and right_flag:
        C_star = np.sqrt(C_star_l*C_star_r)
    elif left_flag:
        C_star = C_star_l
    else:
        C_star = C_star_r

    C_psi = np.zeros((ND,Nt+1),np.float64)
    C_psi[:,0] = 1.0

    rescaled_sols = qnute_sols.copy()

    for Di in range(ND):
        prev_time = 1
        prev_rescale = 0
        for ti in range(1,Nt+1):
            if ti % rescale_frequency != 0:
                C_psi[Di,ti] = np.prod(qnute_norms[Di,prev_time:ti])*C_psi[Di,prev_rescale]
            else:
                C_psi[Di,ti] = C_star[Di,ti//rescale_frequency - 1]
                prev_time = ti+1
                prev_rescale = ti
    C_psi *= qnute_norms[0,0]
    
    for Di in range(ND):
        for ti in range(Nt+1):
            rescaled_sols[Di,ti,:] *= C_psi[Di,ti]
    return rescaled_sols, C_psi

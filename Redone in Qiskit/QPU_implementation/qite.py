import numpy as np
import matplotlib.pyplot as plt
from helper import measure,propogate,estimate_assignment_probs,update_alist

def ansatz():
    None

def measure_energy():
    None

def get_expectation():
    None

def qite_step(alist, shots, qc, qbits, correction_matrix, db, delta, hm_list):
    for j in range(len(hm_list)):
        sigma_expectation = get_expectation()
        norm = update_alist()
    return alist

def qite(qc, qbits, shots, db, delta, N, hm_list):
    E = np.zeros(N+1, dtype=complex)
    alist = []

    # Estimate the readout probabilities p(0|0), p(0|1), p(1|0), p(1|1)
    correction_matrix = estimate_assignment_probs()
    # Record the initial energy of the system
    E[0] = measure_energy()

    # QITE Loop
    for i in range(1,N+1):
        # correction_matrix = estimate_assignment_probs()
        alist = qite_step(alist, shots, qc, qbits, correction_matrix, db, delta, hm_list)
        # Record the energy of the system after the QITE step
        E[i] = measure_energy()

    return E 


if __name__ == '__main__':
    # ---- input parameters for qite
	# Produces Figure 2(e) of https://arxiv.org/pdf/1901.07653.pdf
	N = 25
	shots = 1000
	db = 0.1
	qc = '1q-qvm'
	qbits = [0]
	hm_list = []
	hm_list.append([])
	hm_list[0].append([[1],[1/np.sqrt(2)]])
	hm_list.append([])
	hm_list[1].append([[3],[1/np.sqrt(2)]])
	delta = 0.1
	
	E = qite(qc,qbits,shots,db,delta,N,hm_list)
	plt.plot(np.arange(0,N+1)*db,E,'ro',label='QITE')
	plt.grid()
	plt.show()
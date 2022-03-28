import numpy as np
import matplotlib.pyplot as plt
from helper import run_circuit,measure,propogate,estimate_assignment_probs,update_alist
from helper import sv_sim, aer_sim

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def ansatz():
	None

def measure_energy(alist, hm_list, backend, shots=1024):
	Energy = 0
	Nterms = len(hm_list)
	for i in range(Nterms):
		for hm in hm_list[i]:
			for j in range(len(hm[0])):
				qbit = QuantumRegister(1)
				cbit = ClassicalRegister(1)
				qc = QuantumCircuit(qbit,cbit)
				# ansatz(qc,qbit)
				propogate(qc, alist, qbit)
				Energy += hm[1][j] * measure(qc,hm[0][j],qbit,cbit[0],backend,num_shots=shots)
	return Energy


def get_expectation(alist, backend, shots=1024):
	sigma_expectation = np.zeros([4],dtype=complex)
	
	for j in range(4):
		qbit = QuantumRegister(1)
		cbit = ClassicalRegister(1)
		qc = QuantumCircuit(qbit,cbit)
		# ansatz(qc,qbit)
		propogate(qc, alist, qbit)
		sigma_expectation[j] = measure(qc,j,qbit,cbit[0],backend,num_shots=shots)
		
	return sigma_expectation

def qite_step(alist, db, delta, hm_list, backend, shots=1024):
	for j in range(len(hm_list)):
		sigma_expectation = get_expectation(alist, backend, shots=shots)
		norm = update_alist(sigma_expectation, alist, db, delta, hm_list[j])
	return alist

def qite(db, delta, N, hm_list, backend, shots=1024):
	E = np.zeros(N+1, dtype=complex)
	alist = []

	# Estimate the readout probabilities p(0|0), p(0|1), p(1|0), p(1|1)
	# correction_matrix = estimate_assignment_probs()
	# Record the initial energy of the system
	E[0] = measure_energy(alist, hm_list, backend, shots=shots)

	# QITE Loop
	for i in range(1,N+1):
		# correction_matrix = estimate_assignment_probs()
		alist = qite_step(alist, db, delta, hm_list, backend, shots=shots)
		# Record the energy of the system after the QITE step
		E[i] = measure_energy(alist, hm_list, backend, shots=shots)

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
	
	E = qite(db, delta, N, hm_list, aer_sim, shots=shots)
	plt.plot(np.arange(0,N+1)*db,E,'ro',label='QITE')
	plt.grid()
	plt.show()
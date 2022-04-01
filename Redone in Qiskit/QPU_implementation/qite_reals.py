import numpy as np
import matplotlib.pyplot as plt
from helper import run_circuit,measure_mult,propogate_reals,estimate_assignment_probs,update_alist_reals, odd_y_pauli
from helper import sv_sim, aer_sim
import time

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def ansatz():
	None

def measure_energy(alist, nbits, hm_list, backend, shots=1024):
	Energy = 0
	Nterms = len(hm_list)
	for i in range(Nterms):
		for hm in hm_list[i]:
			qbits = hm[2]
			num_measuring_bits = len(qbits)
			for j in range(len(hm[0])):
				qc = QuantumCircuit(nbits,nbits)
				# ansatz(qc,qbit)
				propogate_reals(qc, alist, qbits, odd_y_pauli(num_measuring_bits))
				Energy += hm[1][j] * measure_mult(qc,hm[0][j],qbits,qbits,backend,num_shots=shots)
	return Energy


def get_expectation(alist, qbits, nbits, backend, shots=1024):
	num_measuring_bits = len(qbits)
	nops = 4**num_measuring_bits
	sigma_expectation = np.zeros([nops],dtype=complex)
	sigma_expectation[0] = 1 # <psi|I|psi> = 1 always
	for j in range(1,nops):
		qc = QuantumCircuit(nbits,nbits)
		# ansatz(qc,qbit)
		propogate_mult(qc, alist, qbits)
		sigma_expectation[j] = measure_mult(qc,j,qbits,qbits,backend,num_shots=shots)
		
	return sigma_expectation

def qite_step(alist, db, delta, nbits, hm_list, backend, shots=1024):
	for j in range(len(hm_list)):
		sigma_expectation = get_expectation(alist,hm_list[j][0][2], nbits, backend, shots=shots)
		norm = update_alist_mult(sigma_expectation, alist, db, delta, hm_list[j])
	return alist

def qite(db, delta, N, nbits, hm_list, backend, shots=1024, debug=False):
	E = np.zeros(N+1, dtype=complex)
	times = np.zeros(N+1)
	alist = []

	# Estimate the readout probabilities p(0|0), p(0|1), p(1|0), p(1|1)
	# correction_matrix = estimate_assignment_probs()
	# Record the initial energy of the system
	E[0] = measure_energy(alist, nbits, hm_list, backend, shots=shots)

	# QITE Loop
	start = 0
	end = 0
	if debug:
		print('Starting QITE Loop')
	for i in range(1,N+1):
		if debug:
			print('i={} -- '.format(i), end='', flush=True)
			start = time.time()

		# correction_matrix = estimate_assignment_probs()
		alist = qite_step(alist, db, delta, nbits, hm_list, backend, shots=shots)
		# Record the energy of the system after the QITE step
		E[i] = measure_energy(alist, nbits, hm_list, backend, shots=shots)

		if debug:
			end = time.time()
			duration = end-start
			times[i] = duration
			print('Execution time: {:.2f} seconds'.format(duration))

	return E,times


if __name__ == '__main__':
	# ---- input parameters for qite
	# Produces Figure 2(e) of https://arxiv.org/pdf/1901.07653.pdf
	N = 25
	shots = 1000
	db = 0.1
	delta = 0.1

	nbits = 2

	hm_list = []
	hm_list.append([])
	hm_list[0].append([[1],[1/2],[0]])
	hm_list.append([])
	hm_list[1].append([[3],[1/2],[1]])
	
	
	E = qite(db, delta, N, nbits, hm_list, aer_sim, shots=shots)
	plt.plot(np.arange(0,N+1)*db,E,'ro',label='QITE')
	plt.grid()
	plt.show()
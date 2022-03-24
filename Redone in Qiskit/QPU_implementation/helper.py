import numpy as np
from index import idx, coeff

from qiskit import execute, Aer
sv_sim = Aer.get_backend('statevector_simulator')
aer_sim = Aer.get_backend('aer_simulator')

def run_circuit(qc, backend, num_shots=1024):
    '''
    Run the circuit and return a dictionary of the measured counts

    qc:         The circuit you are running
    backend:    What backend you're running the circuit on
    num_shots:  The number of times you run the circuit for measurement statistics
    '''
    result = execute(qc, backend, shots=num_shots).result()
    return dict(result.get_counts())

def measure(qc, idx, qbit, cbit, backend, num_shots=1024):
    '''
    Measure qbit into cbit num_shots number of times in the sigma_idx basis and return the expectation
    qc:         The quantum circuit
    idx:        The pauli matrix identifier (0,1,2,3) === (I,X,Y,Z)
    qbit:       The qubit to measure
    cbit:       The classical bit to record the measurement
    backend:    The backend to run the circuit on
    num_shots:  The number of repetitions for measurement statistics
    '''
    if idx == 0:
        # I only has +1 eigenvalues, so the measurement is +1
        return 1
    elif idx == 1:
        # Rotate to X-basis
        qc.h(qbit)
    elif idx == 2:
        # Rotate to Y-basis
        qc.rx(np.pi/2, qbit)
    elif idx == 3:
        # Already in Z-basis
        None
    else:
        raise ValueError('Only 0,1,2,3 are valid idx')
    
    # Add the measurement to the circuit
    qc.measure(qbit, cbit)
    # Get the measurement statistics
    counts = run_circuit(qc, backend, num_shots=num_shots)
    
    # Find the index of cbit in the circuit's classical bits
    index = 0
    for bit in qc.clbits:
        if bit == cbit:
            break
        index += 1

    
    expectation = 0

    # Loop through the counts
    for key in counts.keys():
        k = key.replace(' ', '')
        # Add the +1 eigenvalue measurements
        if k[-(index+1)] == '0':
            expectation += counts[key]
        # Add the -1 eigenvalue measurements
        else:
            expectation -= counts[key]
            
    # Divide by total shots to get the statistical expectation of the measurement
    return float(expectation)/num_shots

def propogate():
    None

def update_alist():
    None

def estimate_assignment_probs():
    None

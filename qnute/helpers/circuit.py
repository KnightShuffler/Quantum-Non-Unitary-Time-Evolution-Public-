import numpy as np
from qiskit import execute
from .pauli import same_pauli_dicts

#-------------------------#
# Quantum Circuit Related #
#-------------------------#
def get_qc_bases_from_pauli_dict(p_dict, qubit_map):
    '''
    describes which basis: X,Y,Z the qubit operation is described in from a
    Pauli string dictionary and a map from qubit coordinates to qubit indices
    '''
    bases = {}
    # print(str(p_dict), str(qubit_map))
    for coord in p_dict.keys():
        gate = p_dict[coord]
        if not isinstance(coord, tuple):
            coord = (coord,)
        qbit = qubit_map[coord]
        if gate != 0:
            bases[qbit] = gate
    return bases

def run_circuit(qc, backend, num_shots=1024):
    '''
    Run the circuit and return a dictionary of the measured counts

    qc:         The circuit you are running
    backend:    What backend you're running the circuit on
    num_shots:  The number of times you run the circuit for measurement statistics
    '''
    result = execute(qc, backend, shots=num_shots).result()
    return dict(result.get_counts())

def measure(qc, p_dict, qubit_map, backend, num_shots=1024):
    '''
    Measures in the pauli string P basis and returns the expected value from the statistics
    assumes the quantum circuit has the same number of quantum and classial bits
    Pauli string is described with a Pauli string dictionary
    '''
    
    # <psi|I|psi> = 1 for all states
    if same_pauli_dicts(p_dict, {}):
        return 1, {}

    # Stores the qubit indices that aren't acted on by I
    bases = get_qc_bases_from_pauli_dict(p_dict, qubit_map)
    active = list(bases.keys())

    # Rotate to Z basis
    for i in active:
        if bases[i] == 1:
            # Apply H to rotate from X to Z basis
            qc.h(i)
        elif bases[i] == 2:
            # Apply Rx(pi/2) to rotate from Y to Z basis
            qc.rx(np.pi/2, i)

    # Add measurements to the circuit
    qc.measure(active, active)

    # Get the measurement statistics
    counts = run_circuit(qc, backend, num_shots=num_shots)

    expectation = 0
    for key in counts:
        # Remove any spaces in the key
        k = key.replace(' ','') 
        # reverse the key to match ordering of bits
        k = k[::-1]
        sign = 1
        # Every 1 measured is a -1 eigenvalue
        for a in active:
            if k[ a ] == '1':
                sign *= -1
        expectation += sign*counts[key]

    expectation /= num_shots
    return expectation, counts

def pauli_string_exp(qc, p_dict, qubit_map, theta):
    '''
    add gates to perform exp(-i theta/2 P) where P is the pauli string
    P is described by a Pauli string dictionary
    '''
    bases = get_qc_bases_from_pauli_dict(p_dict, qubit_map)
    active = list(bases.keys())
    nactive = len(active)
    
    # If the string is identity, the effect is to multiply by a global phase, so we can ignore it
    if nactive == 0:
        return
    # If there is only one active qubit, we can rotate it directly
    elif nactive == 1:
        gate = bases[active[0]]
        if gate == 1:
            qc.rx(theta, active[0])
        elif gate == 2:
            qc.ry(theta, active[0])
        elif gate == 3:
            qc.rz(theta, active[0])
    # Apply the rotation about the Pauli string for multiple active qubits
    else:
        # Rotate to Z basis
        for i in active:
            # Apply H to go from X to Z basis
            if bases[i] == 1:
                qc.h(i)
            # Apply Rx(pi/2) to go from Y to Z basis
            elif bases[i] == 2:
                qc.rx(np.pi/2, i)
        
        # Cascading CNOTs
        for i in range(nactive-1):
            qc.cx(active[i],active[i+1])
            
        # Rotate
        qc.rz(theta, active[-1])
        
        # Undo cascading CNOTs
        for i in range(nactive-2,-1,-1):
            qc.cx(active[i],active[i+1])
            
        # Rotate back to original basis
        for i in active:
            # Apply H to go from Z to X basis
            if bases[i] == 1:
                qc.h(i)
            # Apply Rx(-pi/2) to go from Z to Y basis
            elif bases[i] == 2:
                qc.rx(-np.pi/2, i)

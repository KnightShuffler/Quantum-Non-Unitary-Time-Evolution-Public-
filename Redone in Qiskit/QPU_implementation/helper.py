import numpy as np
from index import idx, coeff

from qiskit import execute, Aer
sv_sim = Aer.get_backend('statevector_simulator')
aer_sim = Aer.get_backend('aer_simulator')

def int_to_base4(x, nbit):
    pauli = [0] * nbit
    for i in range(nbit):
        pauli[i] = x % 4
        x = x//4
    return pauli

def base4_to_int(pauli):
    x = 0
    for i in range(len(pauli)):
        x += pauli[i] * 4**i
    return x

def pauli_prod(p1,p2,nbits):
    pauli1 = int_to_base4(p1,nbits)
    pauli2 = int_to_base4(p2,nbits)
    
    c = 1+0.j
    prod = [0]*nbits
    
    for i in range(nbits):
        prod[i] = idx[pauli1[i], pauli2[i]]
        c *= coeff[pauli1[i], pauli2[i]]
    
    return base4_to_int(prod), c

def pauli_exp(qc, qbits, pauli, theta):
    '''
    adds gates to calculate exp(-i theta/2 p), where p is a Pauli string
    '''
    nbit = len(qbits)

    pstring = int_to_base4(pauli, nbit)
    cnots = []

    # Do a change of basis
    for i in range(nbit):
        if pstring[i] == 1:
            qc.h(qbits[i])
        elif pstring[i] == 2:
            qc.rx(np.pi/2, qbits[i])
        elif pstring[i] == 3:
            None
        else: # pstring[i] == 0, don't add this to the list of cnots
            continue
        cnots.append(qbits[i])
    
    # Apply the cascading CNOTs
    for i in range(len(cnots)-1):
        qc.cx(cnots[i], cnots[i+1])
    
    qc.rz(theta,cnots[-1])

    # Undo the cascading CNOTs
    for i in range(len(cnots)-2,-1,-1):
        qc.cx(cnots[i], cnots[i+1])
    
    # Undo the change of basis
    for i in range(nbit):
        if pstring[i] == 1:
            qc.h(qbits[i])
        elif pstring[i] == 2:
            qc.rx(-np.pi/2, qbits[i])
        else: # do nothing for 0,3
            None




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

def measure_mult(qc, idx, qbits, cbits, backend, num_shots=1024):
    if idx == 0:
        return 1

    nbits = len(qbits)
    ids = int_to_base4(idx, nbits)

    measures = []

    for i in range(nbits):
        if ids[i] == 0:
            # I only has +1 eigenvalues, so the measurement is +1
            continue
        elif ids[i] == 1:
            # Rotate to X-basis
            qc.h(qbits[i])
        elif ids[i] == 2:
            # Rotate to Y-basis
            qc.rx(-np.pi/2, qbits[i])
        elif ids[i] == 3:
            # Already in Z-basis
            None
        else:
            raise ValueError('Only 0,1,2,3 are valid idx')
        measures.append( [qbits[i],cbits[i]] )

    # print(measures)
    # Add measurements to the circuit
    for m in measures:
        qc.measure(m[0],m[1])

    # Get the measurement statistics
    counts = run_circuit(qc, backend, num_shots=num_shots)

    expectation = 0
    for key in counts.keys():
        k = key.replace(' ','') 
        # reverse the key to match ordering of bits
        k = k[::-1]
        sign = 1
        # print('key: ', k)
        for m in measures:
            if k[ m[1] ] == '1':
                sign *= -1
        expectation += sign*counts[key]

    expectation /= num_shots
    return expectation

def propogate(qc, alist, qbit):
    # Circuit to propogate the state
    if len(alist) == 0:
        return
    else:
        for t in range(len(alist)):
            for gate in range(1,4):
                angle = np.real(alist[t][gate])
                if gate == 1:
                    qc.rx(angle, qbit)
                elif gate == 2:
                    qc.ry(angle, qbit)
                elif gate == 3:
                    qc.rz(angle, qbit)
                else:
                    raise ValueError('gate should only take values 1,2,3')

def propogate_mult(qc, alist, qbits):
    nbits = len(qbits)
    for t in range(len(alist)):
        for gate in range(1,4**nbits):
            angle = np.real(alist[t][gate])
            if np.abs(angle) > 1e-5:
                pauli_exp(qc,qbits,gate,angle)

def update_alist(sigma_expectation,alist,db,delta,hm):
	# Obtain A[m]

	# Step 1: Obtain S matrix
	S = np.zeros([4,4],dtype=complex)
	for i in range(4):
		for j in range(4):
			S[i,j] = sigma_expectation[idx[i,j]]*coeff[i,j]

	# Step 2: Obtain b vector
	b = np.zeros([4],dtype=complex)
	c = 1
	for i in range(len(hm[0][0])):
		c -= 2*db*hm[0][1][i]*sigma_expectation[hm[0][0][i]]
	c = np.sqrt(c)
	for i in range(4):
		b[i] += (sigma_expectation[i]/c-sigma_expectation[i])/(db)
		for j in range(len(hm[0][0])):
			b[i] -= hm[0][1][j]*coeff[i,hm[0][0][j]]*sigma_expectation[idx[i,hm[0][0][j]]]/c 
		b[i] = 1j*b[i] - 1j*np.conj(b[i])

	# Step 3: Add regularizer
	dalpha = np.eye(4)*delta

	# Step 4: Solve for linear equation, the solution is multiplied by -2 because of the definition of unitary rotation gates is exp(-i theta/2)
	x = np.linalg.lstsq(S+np.transpose(S)+dalpha,-b,rcond=-1)[0]
	alist.append([])
	for i in range(len(x)):
		alist[-1].append(-x[i]*2*db)
	return c

def update_alist_mult(sigma_expectation, alist, db, delta, hm):
    # number of qubits the hm term acts on
    nbits = len(hm[0][2])
    # number of pauli terms in hm
    nterms = len(hm[0][0])
    # number of pauli strings on that many qubits
    nops = 4**nbits
    
    # Step 1: Obtain S matrix
    S = np.zeros([nops,nops],dtype=complex)
    for i in range(nops):
        for j in range(nops):
            p,c_ = pauli_prod(i,j)
            S[i,j] =  sigma_expectation[p] * c_
            
    # Step 2: Obtain b vector
    b = np.zeros(nops,dtype=complex)
    c = 1
    
    for i in range(nterms):
        c -= 2 * db * hm[0][1][i] * sigma_expectation[hm[0][0][i]]
    c = np.sqrt(c)
    
    for i in range(nops):
        b[i] += ( sigma_expectation[i]/c - sigma_expectation[i] ) / db
        for j in range(nterms):
            p,c_ = pauli_prod(i,hm[0][0][j])
            b[i] -= hm[0][1][j] * c_ * sigma_expectation[p] / c
    b = 1j * (b - np.conj(b))
    
    # Step 3: Add regularizer
    dalpha = np.eye(nops)*delta
    
    # Step 4: Solve for linear equation, the solutions is multiplies by -2 because of the definition of unitary rotations is exp(-i theta/2)
    x = np.linalg.lstsq(S + np.transpose(S) + dalpha, -b, rcond=-1)[0]
    alist.append([])
    for i in range(len(x)):
        alist[-1].append(-x[i]*2*db)
        
    return c

def estimate_assignment_probs():
    None

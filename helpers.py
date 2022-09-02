import numpy as np

from qiskit import execute

import warnings

#-----------------#
# General Helpers #
#-----------------#

# Error Tolerance
TOLERANCE = 1e-5

def int_to_base(x, b, num_digits):
    '''
    convert a non-negative integer x to base b and return the digits in a list
    '''
    if x < 0:
        raise ValueError('x must be non-negative')
    digits = [0] * num_digits
    for i in range(num_digits):
        digits[i] = x % b
        x = x//b
    return digits

def base_to_int(digits, b):
    '''
    convert a list of base b digits to an integer
    '''
    x = 0
    for i in range(len(digits)):
        x += digits[i] * (b**i)
    return x

#----------------------------#
# Manhattan Distance Helpers #
#----------------------------#

def in_lattice(point, d, l):
    '''
    Returns True if the point is within a d-dimensional lattice
    with non-negative coordinates, of side length l
    '''
    for i in range(d):
        if point[i] < 0 or point[i] >= l:
            return False
    return True

def get_center(points):
    '''
    Returns the coordinates of the center of the points
    '''
    n = len(points)
    return np.sum(points,axis=0)/n

def manhattan_dist(a, b):
    '''
    Returns the Manhatan distance between points a and b
    '''
    return np.sum(np.abs(a - b))

def within_radius(center, point, radius):
    '''
    Returns True if the point is within a Manhattan distance of radius 
    away from the center
    '''
    return manhattan_dist(point, center) <= radius

def get_m_sphere(c, R, d, l):
    '''
    Returns the list of points at a Manhattan distance of R from the point c
    in a d-dimensional lattice of side length l
    '''
    sphere = []
    # Cast c to a tuple, important for the d=1 base case
    try:
        c = tuple(c)
    except TypeError as e:
        if type(c) is not tuple:
            c = (c,)

    lb = max(int(np.ceil(c[0] - R)), 0)
    ub = min(int(np.ceil(c[0] + R)), l-1)
    for i in np.arange(lb, ub+1, 1):
        if d > 1:
            # calculate new radius
            if i <= c[0]:
                r = i - (c[0] - R)
            else:
                r = R - (i - c[0])

            sub_sphere = get_m_sphere(c[1:], r, d-1, l)
            for point in sub_sphere:
                sphere.append( (i,) + point )
        else:
            sphere.append( (i,) )
    return sphere

def min_bounding_sphere(points):
    center = get_center(points)
    max_rad = 0
    for p in points:
        rad = manhattan_dist(p, center)
        if rad > max_rad:
            max_rad = rad
    return center, max_rad

#----------#
# Sampling #
#----------#
def sample_from_a(a):
    '''
    Given a vector of real numbers a,
    returns a vector where all elements are 0 except for a random
    index i with probability:
        P(i) = |a[i]|/sum(|a|)
    and that index has the original a[i] in the output
    '''

    cdf = np.abs(a.copy())  # load |a|
    cdf = np.cumsum(cdf)    # Convert to CDF
    
    # Generate a random number
    y = np.random.uniform(0.0,cdf[-1])

    # Flag to see if the index i was found
    found_flag = False 

    for i in range(len(cdf)):
        if y <= cdf[i] and not found_flag:
            found_flag = True
            cdf[i] = a[i]
        else:
            cdf[i] = 0.0
    
    return cdf

#----------------------------#
# Pauli/Sigma Matrix Related #
#----------------------------#

sigma_matrices = np.zeros([4,2,2], dtype=complex)
# sigma_matrices is a list of 4x4 matrices, sigma_matrix[0] is I, etc up to Z
sigma_matrices[0] = np.array([[1,0],[0,1]])
sigma_matrices[1] = np.array([[0,1],[1,0]])
sigma_matrices[2] = np.array([[0,-1j],[1j,0]])
sigma_matrices[3] = np.array([[1,0],[0,-1]])

# maps the product of the Pauli Matrices
# sigma_i sigma_j = pauli_prod[1][i,j] * sigma_(pauli_prod[0][i,j])
#                    ^ coefficient               ^ matrix identifier
pauli_prod = [ np.zeros([4,4],dtype=int), np.zeros([4,4],dtype=complex) ]

pauli_prod[0][0,:]=[0,1,2,3]
pauli_prod[0][1,:]=[1,0,3,2]
pauli_prod[0][2,:]=[2,3,0,1]
pauli_prod[0][3,:]=[3,2,1,0]

pauli_prod[1][0,:]=[1,  1,  1,  1]
pauli_prod[1][1,:]=[1,  1, 1j,-1j]
pauli_prod[1][2,:]=[1,-1j,  1, 1j]
pauli_prod[1][3,:]=[1, 1j,-1j,  1]

def pauli_string_prod(p1,p2,nbits):
    '''
    returns the product of two nbit long pauli strings
    p1 and p2 are both nbit digit base 4 number representing the pauli string
    
    an nbit digit base 4 number representing the product, and the coefficient are returned
    '''
    pstring1 = int_to_base(p1,4,nbits)
    pstring2 = int_to_base(p2,4,nbits)
    
    c = 1+0.j
    prod = [0]*nbits
    
    for i in range(nbits):
        prod[i] = pauli_prod[0][pstring1[i], pstring2[i]]
        c      *= pauli_prod[1][pstring1[i], pstring2[i]]
    
    return base_to_int(prod,4), c

def odd_y_pauli_strings(nbits):
    '''
    returns a list of all the nbit long pauli strings with an odd number of Ys
    '''
    
    nops = 4**nbits
    odd_y = []
    for i in range(nops):
        p = int_to_base(i,4,nbits)
        num_y = 0
        # count the number of Y gates in the string
        for x in p:
            if x == 2:
                num_y += 1
        # append if odd number of Ys
        if num_y %2 == 1:
            odd_y.append(i)
    return odd_y

def ext_domain_pauli(p, active, domain):
    '''
    returns the id of the pauli string p that acts on qubits in active, when the domain is extended to domain
    '''
    pstring = int_to_base(p,4,len(active))
    new_pstring = [0] * len(domain)
    for i in range(len(active)):
        if active[i] not in domain:
            print('Error Occured in ext_domain_pauli:')
            print('active:', active)
            print('domain:',domain)
            print('active[{}] not in domain'.format(i))
        ind = domain.index(active[i])
        new_pstring[ind] = pstring[i]
    return base_to_int(new_pstring, 4)


#-------------------------#
# Quantum Circuit Related #
#-------------------------#

def run_circuit(qc, backend, num_shots=1024):
    '''
    Run the circuit and return a dictionary of the measured counts

    qc:         The circuit you are running
    backend:    What backend you're running the circuit on
    num_shots:  The number of times you run the circuit for measurement statistics
    '''
    result = execute(qc, backend, shots=num_shots).result()
    return dict(result.get_counts())

def measure(qc, p, qbits, backend, num_shots=1024):
    '''
    Measures in the pauli string P basis and returns the expected value from the statistics
    assumes the quantum circuit has the same number of quantum and classial bits
    '''
    
    # <psi|I|psi> = 1 for all states
    if p == 0:
        return 1, {}
    
    nbits = len(qbits)
    pstring = int_to_base(p, 4, nbits)

    # Stores the qubit indices that aren't acted on by I
    active = []

    for i in range(nbits):
        if pstring[i] == 0:
            # I only has +1 eigenvalues, so the measurement is +1
            continue
        elif pstring[i] == 1:
            # Apply H to rotate from X to Z basis
            qc.h(qbits[i])
        elif pstring[i] == 2:
            # Apply Rx(-pi/2) to rotate from Y to Z basis
            qc.rx(-np.pi/2, qbits[i])
        elif pstring[i] == 3:
            # Already in Z-basis
            None
        active.append( qbits[i] )

    # Add measurements to the circuit
    for bit in active:
        qc.measure(bit,bit)

    # Get the measurement statistics
    counts = run_circuit(qc, backend, num_shots=num_shots)

    expectation = 0
    for key in counts.keys():
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

def pauli_string_exp(qc, qbits, p, theta):
    '''
    add gates to perform exp(-i theta/2 P) where P is the pauli string
    '''
    
    nbits = len(qbits)
    
    pstring = int_to_base(p,4,nbits)
    
    # stores the qubit indices that aren't I
    active = []
    # stores the corresponding gate on the active qubit
    gates  = []
    for i in range(nbits):
        if pstring[i] != 0:
            active.append(qbits[i])
            gates.append(pstring[i])
    
    nactive = len(active)
    
    # If the string is identity, the effect is to multiply by a global phase, so we can ignore it
    if nactive == 0:
        return
    # If there is only one active qubit, we can rotate it directly
    elif nactive == 1:
        if gates[0] == 1:
            qc.rx(theta, active[0])
        elif gates[0] == 2:
            qc.ry(theta, active[0])
        elif gates[0] == 3:
            qc.rz(theta, active[0])
    # Apply the rotation about the Pauli string for multiple active qubits
    else:
        # Rotate to Z basis
        for i in range(nactive):
            # Apply H to go from X to Z basis
            if gates[i] == 1:
                qc.h(active[i])
            # Apply Rx(pi/2) to go from Y to Z basis
            elif gates[i] == 2:
                qc.rx(np.pi/2, active[i])
        
        # Cascading CNOTs
        for i in range(nactive-1):
            qc.cx(active[i],active[i+1])
            
        # Rotate
        qc.rz(theta, active[-1])
        
        # Undo cascading CNOTs
        for i in range(nactive-2,-1,-1):
            qc.cx(active[i],active[i+1])
            
        # Rotate back to original basis
        for i in range(nactive):
            # Apply H to go from Z to X basis
            if gates[i] == 1:
                qc.h(active[i])
            # Apply Rx(-pi/2) to go from Z to Y basis
            elif gates[i] == 2:
                qc.rx(-np.pi/2,active[i])

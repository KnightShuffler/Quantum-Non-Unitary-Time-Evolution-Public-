import numpy as np
from . import int_to_base
from . import base_to_int
from . import TOLERANCE


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

def get_pauli_prod_matrix(p,nbits):
    pdigits = int_to_base(p, 4, nbits)
    term_mat = np.ones((1,1),dtype=np.complex128)
    for p in pdigits:
        term_mat = np.kron(sigma_matrices[p], term_mat)
    return term_mat

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

def pauli_index_to_dict(p, domain):
    '''
    Converts a base-4 integer Pauli string index p to a dictionary of which Pauli
    operators act on each qubit in the domain
    '''
    p_dict = {}
    p_ops = int_to_base(p, 4, len(domain))
    for i in range(len(domain)):
        p_dict[domain[i]] = p_ops[i]
    return p_dict

def same_pauli_dicts(pd1, pd2):
    '''
    Returns whether two Pauli string dictionaries represent the same operator
    '''
    keys = list( set(pd1.keys()) | set(pd2.keys()) )
    for key in keys:
        if key not in pd1.keys():
            if pd2[key] != 0:
                return False
        elif key not in pd2.keys():
            if pd1[key] != 0:
                return False
        else:
            if pd1[key] != pd2[key]:
                return False
    return True

def pauli_dict_product(p1_dict, p2_dict):
    '''
    Returns the product of two Pauli string dictionaries and the phase accumulated
    '''
    keys = list( set(p1_dict.keys()) | set(p2_dict.keys()) )
    prod_dict = {}
    coeff = 1+0.j
    for key in keys:
        if key not in p1_dict.keys():
            p1_dict[key] = 0
        if key not in p2_dict.keys():
            p2_dict[key] = 0
        
        coeff *= pauli_prod[1][p1_dict[key], p2_dict[key]]
        prod_dict[key] = pauli_prod[0][p1_dict[key], p2_dict[key]]
    return prod_dict, coeff

def get_full_pauli_product_matrix(partial_pstring, active, nbits):
    '''
    Returns the full matrix of the the Pauli product index p acting on 
    qubits indexed in active in system with number of qubits=nbits
    '''
    nactive = len(active)
    assert len(partial_pstring) == nactive, 'partial_pstring must have the same length as active'
    # partial_pstring = int_to_base(p, 4, nactive)
    full_pstring = [0] * nbits
    for k in range(nactive):
        full_pstring[active[k]] = partial_pstring[k]
    p_mat = sigma_matrices[full_pstring[0]]
    for k in range(1,nbits):
        p_mat = np.kron(sigma_matrices[full_pstring[k]], p_mat)
    return p_mat

def get_pauli_eigenspace(partial_pstring, active, nbits, eigval):
    w,v = np.linalg.eig(get_full_pauli_product_matrix(partial_pstring, active, nbits))
    ind = np.where(np.abs(w-eigval) < TOLERANCE)[0]
    return v[:,ind].T

def pauli_str_to_index(p):
    ind = 0
    #power = 0
    for (power, char) in enumerate(p):
        if char != 'I':
            ind += (4**power) * (char-ord('X')+1)
    return ind
import numpy as np
import scipy
from binary_functions import Int2Bas, Bas2Int, Opp2Str, Psi2Str, Lst2Str

#------------------#

# Encodes the product of Pauli matrices, first array is the new matrix, second is the phase associated
pauli_product=[np.zeros((4,4),dtype=int),np.zeros((4,4),dtype=complex)]

# pauli_product[0][p][q] maps which gate q maps to when acted on by p from the left
pauli_product[0][0,:]=[0,1,2,3]
pauli_product[0][1,:]=[1,0,3,2]
pauli_product[0][2,:]=[2,3,0,1]
pauli_product[0][3,:]=[3,2,1,0]

# pauli_product[1][p][q] maps the accumulated phase when p acts on q from the left
pauli_product[1][0,:]=[1,  1,  1,  1]
pauli_product[1][1,:]=[1,  1, 1j,-1j]
pauli_product[1][2,:]=[1,-1j,  1, 1j]
pauli_product[1][3,:]=[1, 1j,-1j,  1]

# Example: XY = iZ, so pauli_product[0,1][1][2] = [3, 1.0j]

# Stores the Pauli matrices first two indices are the [i][j] elements of the matrices
# the third index marks which Pauli matrix it is 0,1,2,3 == I,X,Y,Z
sigma_matrices = np.zeros((2,2,4), dtype=complex)
for i in range(2):
	j = (i+1)%2
	sigma_matrices[i,i,0] = 1.0
	sigma_matrices[i,j,1] = 1.0
	sigma_matrices[i,j,2] = 1.0j*(-1.0)**(i+1.0)
	sigma_matrices[i,i,3] = (-1.0)**i

# returns 1 if t==1,2 mod 3
d12 = lambda t: 1 if t%3>0 else 0
d12f = np.vectorize(d12)

# returns 1 if t==2
d2 = lambda t: 1 if t==2 else 0
d2f = np.vectorize(d2)

# returns 1 if t>1
d23 = lambda t: 1 if t>1 else 0
d23f = np.vectorize(d23)

#------------------#

# prints the computational basis kets of nbit_ qubits
def computational_basis(nbit_):
	N=2**nbit_
	for i in range(N):
		print(i,Psi2Str(Int2Bas(i,2,nbit_)))

# prints the Pauli basis operators of nbit_ qubits
def pauli_basis(nbit_):
	M=4**nbit_
	for i in range(M):
		print(i,Opp2Str(Int2Bas(i,4,nbit_)))

#------------------#

# ARGS:
#       active_: a list containing the indices [0 - (nbit_ - 1)] on which the Pauli gates act on
#       nbit_:   the number of qubits
# OUTPUTS:
#       ind_sx, gmm_sx: (4**nact, 2**nbit) matrix
#           - The mth row of th matrix codifies the mth Pauli string (convert m 
#             to a base 4 integer then map 0-3 to IXYZ)
#           - The [m][n] element of ind_sx stores which basis vector the ket |n> 
#             got mapped to under the Pauli string m. (map to a binary int for basis ket)
#           - The [m][n] element of gmm_sx stores the phase of the mapped ket under action of the
#             mapped string


def pauli_action(active_,nbit_,verbose=False):
	nact = len(active_)
	N    = 2**nbit_
	M    = 4**nact

	dot    = [2**(nbit_-1-i) for i in range(nbit_)]
	ind_sx = np.zeros((M,N),dtype=int)
	gmm_sx = np.zeros((M,N),dtype=complex)+1

	# Populate svec with the all possible Pauli strings acting on all the qubits
	svec = np.zeros((M,nbit_),dtype=int) 
	for mu in range(M):
		svec[mu,active_]=Int2Bas(mu,4,nact)
	
	# Marks the presence of X and Y gates in the Pauli strings
	sxyvec = d12f(svec)
	# Marks the presence of Y and Z gates in the Pauli strings
	syzvec = d23f(svec)

	# Populate nyvec with the number of Y gates for each pauli string
	nyvec  = d2f(svec)
	nyvec  = np.einsum('ab->a',nyvec)

	# Populates the basis state bit strings in xvec
	xvec = np.zeros((N,nbit_),dtype=int)
	for xi in range(N):
		xvec[xi,:] = np.asarray(Int2Bas(xi,2,nbit_))

	# Count the number (-1) phases accumulated by the Y,Z gates in the Puali string (only if that qubit is |1>)
	gmm_sx=np.einsum('am,bm->ba',xvec,syzvec)+0j
	# Save the phases
	gmm_sx[:,:]=(-1)**gmm_sx[:,:]
	# For each pauli string 
	for mu in range(M):
		# Multiply the (i) phases accumulated by the Y gates (number of Y gates in the string)
		gmm_sx[mu,:] *= 1j**nyvec[mu]
		# Applies the bit-flips from the X,Y gates in the Pauli string to the basis kets
		yvec          = (xvec[:,:]+sxyvec[mu,:])%2
		# Saves the remapped basis kets as integers in ind_sx
		ind_sx[mu,:]  = np.einsum('a,ba->b',dot,yvec)

	return ind_sx,gmm_sx 
import numpy as np
from qnute.hamiltonian import Hamiltonian
from qnute.hamiltonian.laplacian import generateLaplaceHamiltonian1D

def main():
    # hml1 = [
    #     [np.array([5, 10],dtype=np.uint32), np.array([0.1, 0.1],dtype=np.complex128), [(2,),(3,)]],
    #     [np.array([3], dtype=np.uint32), np.array([0.01],dtype=np.complex128), [(2,)]]
    # ]

    # hml2 = [
    #     [np.array([15, 6],dtype=np.uint32), np.array([0.1, 0.1],dtype=np.complex128), [(0,),(1,)]],
    #     [np.array([2],dtype=np.uint32), np.array([0.01,],dtype=np.complex128), [(0,)]]
    # ]
    n = 3
    qubit_map = {(i,):i for i in range(n)}

    # hml12 = hm_list_tensor(hml1, hml2)
    # hml21 = hm_list_tensor(hml2, hml1)

    # H1 = Hamiltonian(hml1,qubit_map)
    # H2 = Hamiltonian(hml2,qubit_map)
    # H1_H2 = Hamiltonian(hml12, qubit_map)
    # H2_H1 = Hamiltonian(hml21, qubit_map)

    # print(H1)
    # print(H2)
    # print(H1_H2)
    # print(H2_H1)

    H = generateLaplaceHamiltonian1D(n, 0.1, False, False)
    print(H)
    print(H.pterm_list)
    print(H.hm_indices)
    print(np.real(H.get_matrix()))

    # hml1 = [
    #     [np.array([0],dtype=np.uint32), np.array([1.0+0.j],dtype=np.complex128), [1]],
    #     [np.array([3],dtype=np.uint32), np.array([1.0+0.j],dtype=np.complex128), [2]]
    # ]

    # hml2 = [
    #     [np.array([1,2],dtype=np.uint32), np.array([0.5,0.5],dtype=np.complex128), [0]]
    # ]

    # H1 = Hamiltonian(hml1, 3)
    # H2 = Hamiltonian(hml2, 1)

    # print(Hamiltonian.tensor_product(H1,H2))

if __name__ == '__main__':
    main()
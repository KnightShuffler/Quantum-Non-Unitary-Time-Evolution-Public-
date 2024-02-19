import numpy as np
# from .laplace_hamiltonian import hm_list_tensor, get_lowerLeft_hm_list, get_upperRight_hm_list, hm_list_sum, get_laplace1D_hm_list
from .laplace_hamiltonian import generateLaplaceHamiltonian1D
from qnute.hamiltonian import Hamiltonian

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

    hm_list = generateLaplaceHamiltonian1D(n, qubit_map, 0.1, True, True)
    print(hm_list)
    H = Hamiltonian(hm_list, qubit_map)
    print(H)
    print(H.hm_indices)
    print(np.real(H.get_matrix()))

if __name__ == '__main__':
    main()
# Quantum Non-Unitary Time Evolution

Numerical and Qiskit implementations of the Quantum Non-Unitary Time Evolution (QNUTE) algorithm. This is an extension of the [Quantum Imaginary Time Evolution algorithm](https://www.nature.com/articles/s41567-019-0704-4) that allows us to simulate the dynamics of any linear PDE of the form
$$\frac{\partial}{\partial t} f(\vec{x},t) = \hat{H} f(\vec{x},t)$$
where the effective Hamiltonian $\hat{H}$, can be an arbitrary, time-independent linear differential operator. 

## How to Run
To run the simulation of the QNUTE algorithm, run the `qnute()` method defined in `qnute.py`. 
Arguments:
* `params` - A `QNUTE_params` object containing the run parameters for the QNUTE experiment. Details given below.
* `log_to_console` - Set `True` to display time logs of the QNUTE run in the terminal. Default `True`.

Both functions return the values:
* `times` - A list containing the time to complete simulating each time step
* `svs` - A 2-d numpy array containing the list of the state vectors calculated at each simulated time step.
* `a_list` - A list of all the vectors $\vec{a}$ calculated when solving the linear systems $S\vec{a}=\vec{b}$ in each Trotter step.
* `S_list` - A list of all the matrices $S$ for the linear systems generated in each Trotter step.
* `b_list` - A list of all the vectors $\vec{b}$ for the linear systems generated in each Trotter step.
* `c_list` - A list of all the norms $c=\|e^{\hat{H}_m\tau}|\psi\rangle\|$ calculated in the simulation. The product of all these norms should approximate the final norm $\|e^{\hat{H}t}|\psi_0\rangle\|$.

## Effective Hamiltonian Description
Defined in `hamiltonians.py`, the `Hamiltonian` class provides a description of the effective Hamiltonian as a linear combination of Pauli operator tensor products.

It is assumed that the qubits are connected in a $d$-dimensional cubic lattice of size $l$, i.e. each qubit can be indexed with $d$ integer coordinates $(x_1,\cdots,x_d)$ where $0\leq x_i < l$. A mapping between the lattice coordinates and the physical qubit indices of the quantum computer must be provided so the quantum circuits can be generated accordingly. For $d=1$, if a map is not provided, the program automatically maps the lattice coordinate to the same physical qubit index.

The Hamiltonian constructor takes 4 inputs:
* `hm_list` - The Hamiltonian is described as a sum of terms $\hat{H}=\sum_m\hat{H}_m$, where each $\hat{H}_m$ acts on distinct subsets of the qubits in the lattice. Each of these $\hat{H}_m$ terms is described using a list `hm` containing 3 lists:
    * `hm[2]` - List of qubits that $\hat{H}_m$ acts on. This list contains the lattice coordinates of all of the qubits that $\hat{H}_m$ acts on. The coordinates must be tuples of integers. The order of the coordinates in this list matters.
    * `hm[0]` - List of Pauli operator products. This list contains integers corresponding to Pauli operator tensor products when converted to base-4 integers. The digits 0,1,2,3 corresponding to the Pauli operators $\hat{I},\hat{X},\hat{Y},\hat{Z}$ respectively. For example, the Pauli operator product
    $\hat{X}\_{0} \otimes \hat{Y}\_{1} \otimes \hat{Z}\_{2}$
    corresponds to the integer
    $$(321)\_{4} = 3\cdot 4^2 + 2\cdot 4^1 + 1 \cdot 4^0 =(57)\_{10}.$$
    The subscripts $0,1,2$ here correspond to the qubit coordinates `hm[2][0], hm[2][1], hm[2][3]`.
    * `hm[1]` - List of linear combination coefficients of Pauli operators. $\hat{H}_m = \sum_i \alpha_i \hat{\sigma}_i.$ For each Pauli operator product `hm[0][i]`$\equiv\hat{\sigma}_i$, the complex coefficient in the linear combination is provided in this list, `hm[1][i]`$=\alpha_i$.
* `lattice_dim` - The dimension $d$ of the qubit lattice.
* `lattice_bound` - The size $l$ of the lattice.
* `qubit_map` - A dictionary mapping the lattice coordinates to physical qubit indices. Is `None` by default, but this is only valid for $d=1$.

For example, consider a 2-dimensional qubit lattice of size 2, with the lattice mapping shown earlier. The effective Hamiltonian
$$\hat{H} = \left[\hat{Z}\_{(0,0)}\otimes\hat{X}\_{(1,1)} - \hat{X}\_{(0,0)}\otimes\hat{Z}\_{(1,1)}\right] + \left[i\ \hat{I}\_{(0,1)}\otimes\hat{Y}\_{(1,0)}\right]$$
is represented by
```
H = Hamiltonian(
    hm_list = [
        [ [7, 13], [1.0, -1.0], [(0,0), (1,1)] ],
        [ [8],     [1.0j],      [(0,1), (1,0)] ]
    ],
    lattice_dim = 2,
    lattice_bound = 2,
    qubit_map = {(0,0):0, (0,1):1, (1,0):2, (1,1):3}
)
```

## QNUTE Parameters
Defined in `qnute_params.py`, the `QNUTE_params` class contains all of the parameters that QNUTE needs to run an experiment. The constructor takes a `Hamiltonian` object as input, and it will perform all the necessary pre-calculations when the user calls the methods `load_hamiltonian_params()` and `set_run_params()`.

### Hamiltonian Parameters
`QNUTE_params.load_hamiltonian_params()`
For each term $e^{\hat{H}_m \tau}$, in the Trotter product approximation QNUTE calculates a Hermitian operator $\hat{A}$ that specifies the normalized evolution
$$e^{-i\hat{A}\tau} |\psi\rangle \approx \frac{e^{\hat{H}_m\tau}|\psi\rangle}{\|e^{\hat{H}_m\tau}|\psi\rangle\|}.$$

QNUTE allows for approximations making use of the lattice topology of the simulated qubits by shrinking the domain of $\hat{A}$ to a (Manhattan distance) sphere of diameter $D$ around the center of the original domain of $\hat{H}_m$.
Parameter|Description|Default
---------|-----------|-------
`D` | Maximum diameter of the unitary domains.| -
`reduce_dim` | `True` if it is known that the state vector and Hamiltonian terms are all real-valued in the $\hat{Z}$ basis. This roughly halves the dimensionality of the system of linear equations to solve for $\hat{A}$. | `False`
`load_measurements` | Set `True` to pre-calculate which Pauli measurements are required for each $\hat{H}_m$ to generate $\hat{A}$. Set `True` for Qiskit implementation. | `True`

### Run Parameters
`QNUTE_params.set_run_params()`
Parameter|Description|Default
------------|--------|-----
`dt`        | Size of time step $\tau$ | -
`delta`     | Size of regularizer when solving linear equations | - 
`N`         | Number of time steps $t=N\tau$ | - 
`num_shots` | Number of shots used to generate the measurement statistics for each measurement. Irrelevant for numerical implementation. | - 
`backend`   | Which Qiskit backend to use. Currently only the statevector simulator is supported. Irrelevant for numerical implementation. | - 
`init_circ` | A `qiskit.QuantumCircuit` object that generates the initial state from all qubits initialized to $\|0\rangle$. Leave as `None` if using numerical implementation or specifying initial state vector. | `None`
`init_sv`   | Initial state vector for the QNUTE run. If both `init_circ` and `init_sv` are left as `None`, all qubits will be initialized to $\|0\rangle$. | `None`
`store_state_vector` | Set `True` to store the state vector during the simulation. Set `Fasle` when sending to real quantum computers or to not store the state vector in Qiskit simulations. The current implementation does not support `False`.| `True`
`taylor_norm_flag` | Set `True` to calculate the norm of the non-unitary time step with a Taylor series. `False` for real quantum hardware. | `False`
`taylor_truncate_h` | How many terms should be included in the Taylor series expansion of $e^{\hat{H}_m\tau}\|\psi\rangle$ for the numerical simulation. Set to -1 to keep adding terms until the norm of the last term is less than `helpers.TOLERANCE`. | `-1`
`taylor_truncate_a` | How many terms should be included in the Taylor series expansion of $e^{-i\hat{A}\tau}\|\psi\rangle$ for the numerical simulation. Set to -1 to keep adding terms until the norm of the last term is less than `helpers.TOLERANCE`. | `-1`
`trotter_flag` | Set `True` to use a Trotter product to calculate the time evolution operator in the numerical simulation, and `False` to use the Taylor series instead. | `False`


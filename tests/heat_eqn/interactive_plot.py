import numpy as np
from qnute.hamiltonian.finite_difference.laplacian import generateLaplaceHamiltonian1D, generateGrayCodeLaplacian1D
from qnute.simulation.numerical_sim import qnute
from qnute.simulation.numerical_sim import get_theoretical_evolution as get_qnute_th_evolution
from qnute.simulation.parameters import QNUTE_params as Params
from qnute.simulation.output import QNUTE_output as Output
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import UnivariateSpline

from .run_test import get_theoretical_evolution, graycode_permute_matrix

def main():
    n = 4
    qubit_map = {(i,):(i) for i in range(n)}
    assert n-1 > 1

    sv_sample_indices = np.arange(2**(n-1))
    sv_extra_indices = np.array([i for i in range(2**n) if i not in sv_sample_indices])
    homogeneous_flag = False
    graycode_flag = True
    full_circle_flag = False
    reduce_dim_flag = True
    periodic_bc_flag = False

    Nx = 2**n
    dx = 0.1
    L = dx * (Nx+1)
    if homogeneous_flag:
        L = dx * (Nx//2 + 1)
    T = 0.5

    dtau = 0.1
    dt = dtau*dx*dx
    Nt = np.int32(np.ceil(T/dt))

    print(f'Nt = {Nt}, dt = {dt:0.5f}')

    delta = 0.1
    num_shots=0
    backend=None
    trotter_flag = False

    if not graycode_flag:
        H = generateLaplaceHamiltonian1D(n, 0, 1.0, periodic_bc_flag, homogeneous_flag)
    else:
        H = generateGrayCodeLaplacian1D(n, 1.0, periodic_bc_flag, homogeneous_flag)
    # print(H)
    # print(H.pterm_list)
    # print(H.hm_indices)
    # print(np.real(H.get_matrix()))

    P_n = graycode_permute_matrix(n)
    iP_n = np.linalg.inv(P_n)

    times = np.arange(Nt+1)*dt
    x = np.arange((Nx if not homogeneous_flag else Nx//2)+2)*dx
    f = np.zeros(x.shape,dtype=np.complex128)

    freq = 1

    # theoretical_solution = get_theoretical_evolution(L,Nx,dx,Nt,dt,
    #                           sv_sample_indices,
    #                           homogeneous_flag, freq)
    # psi0 = theoretical_solution[0].copy()
    psi0 = np.random.uniform(0.0, 1.0, Nx)
    psi0 /= np.linalg.norm(psi0)
    if graycode_flag:
        psi0 = np.dot(P_n, psi0)
    eig_vals,eig_states = H.get_spectrum()

    params = Params(H,1,n,qubit_map)

    qnute_svs = np.zeros((n-1, times.shape[0], Nx), dtype=np.complex128)
    eig_fids = np.zeros((n-1, times.shape[0], Nx), dtype=np.complex128)

    for Di,D in enumerate(range(2,n+1)):
        if full_circle_flag:
            u_domains = [[j%n for j in range(i,i+D)] for i in range(n)] if D < n else [list(range(n))]
        else:
            u_domains = [list(range(i,i+D)) for i in range(n-D+1)]
        print('u_domains:', u_domains)

        params.load_hamiltonian_params(D, u_domains, reduce_dim_flag, True)
        params.set_run_params(dtau, delta, Nt, num_shots, backend, init_sv=psi0,trotter_flag=trotter_flag)

        out = qnute(params,log_frequency=100,c0=1.0)
        # print(len(out.c_list))

        # print('Final State:')
        # print(out.svs[-1,:])
        qnute_svs[Di,:,:] = out.svs
        for t in range(Nt+1):
            if graycode_flag:
                qnute_svs[Di,t,:] = np.matmul(iP_n, qnute_svs[Di,t,:])
            for i in range(Nx):
                eig_fids[Di,t,i] = np.vdot(eig_states[:,i], out.svs[t,:])
    
    fig, axs = plt.subplots(2, n-1, sharex=True,sharey=True, figsize=((n-1)*4, 2*4))
    plt.subplots_adjust(left=0.15)
    ax_time = plt.axes([0.02, 0.2, 0.03, 0.65], facecolor='lightgoldenrodyellow')
    slider_time = Slider(ax_time, 'Time', 0, Nt, valinit=0, valstep=10, orientation='vertical')
    
    fig.suptitle(f'QITE States at time t={0.0:0.3f}')
    lines = [[[i,i] for i in range(n-1)] for j in range(2)]
    splines = [[[i,i] for i in range(n-1)] for j in range(2)]
    x = np.arange(Nx)
    x_spline = np.linspace(0,Nx-1,1000)
    axs[0,0].set_xticks(x)
    axs[0,0].set_ylim([-1.01,1.01])
    for Di,D in enumerate(range(2,n+1)):
        axs[0,Di].set_title(f'D={D} computational basis')
        axs[1,Di].set_title(f'D={D} eigen basis')
        axs[0,Di].grid(True)
        axs[1,Di].grid(True)

        # lines[0][Di][0], = axs[0,Di].plot(x, qnute_svs[Di,0,:].real)
        lines[1][Di][0], = axs[1,Di].plot(x, eig_fids[Di,0,:].real)
        # lines[0][Di][1], = axs[0,Di].plot(x, qnute_svs[Di,0,:].imag)
        lines[1][Di][1], = axs[1,Di].plot(x, eig_fids[Di,0,:].imag)

        r_spline = UnivariateSpline(x,qnute_svs[Di,0,:].real,s=3)
        i_spline = UnivariateSpline(x,qnute_svs[Di,0,:].imag,s=3)

        splines[0][Di][0], = axs[0,Di].plot(x_spline, r_spline(x_spline))
        splines[0][Di][1], = axs[0,Di].plot(x_spline, i_spline(x_spline))
        



    def update(val):
        ti = slider_time.val
        t = slider_time.val * dt
        fig.suptitle(f'QITE States at time t={t:0.3f}')
        for Di in range(n-1):
            # lines[0][Di][0].set_ydata(qnute_svs[Di,ti,:].real)
            lines[1][Di][0].set_ydata(eig_fids[Di,ti,:].real)
            # lines[0][Di][1].set_ydata(qnute_svs[Di,ti,:].imag)
            lines[1][Di][1].set_ydata(eig_fids[Di,ti,:].imag)

            r_spline = UnivariateSpline(x,qnute_svs[Di,ti,:].real,s=3)
            i_spline = UnivariateSpline(x,qnute_svs[Di,ti,:].imag,s=3)

            splines[0][Di][0].set_ydata(r_spline(x_spline))
            splines[0][Di][1].set_ydata(i_spline(x_spline))
        fig.canvas.draw()
    
    slider_time.on_changed(update)


    plt.show()

if __name__ == "__main__":
    main()
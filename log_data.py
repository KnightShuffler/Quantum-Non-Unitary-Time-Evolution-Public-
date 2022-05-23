import numpy as np
import matplotlib.pyplot as plt

import hamiltonians

def plot_data(fig_title, run_id, params, E, statevectors, eig_flag, prob_flag):
    plt.clf()

    if prob_flag:
        fig,axs = plt.subplots(1,2, figsize=(12,5), sharex=True)
        energy_plot = axs[0]
        prob_plot = axs[1]

        energy_plot.set_title('Mean Energy in QITE')
        prob_plot.set_title('Ground State Probability in QITE')
    else:
        fig,axs = plt.subplots(1,1,figsize=(6,5))
        energy_plot = axs
    
    fig.suptitle(fig_title, fontsize=16)
    plt.subplots_adjust(top=0.85)
    
    energy_plot.plot(np.arange(params.N+1)*params.db, E, 'ro-', label='Mean Energy of State')
    if eig_flag:
        w,v = hamiltonians.get_spectrum(params.hm_list, params.nbits)
        for eig in w:
            eig_line = energy_plot.axhline(y=eig.real, color='k', linestyle='--')
        eig_line.set_label('Hamiltonian Energy Levels')
    
    energy_plot.set_xlabel('Imaginary Time')
    energy_plot.set_ylabel('Energy')
    energy_plot.grid()
    energy_plot.legend(loc='best')

    if prob_flag:
        if not eig_flag:
            w,v = hamiltonians.get_spectrum(params.hm_list, params.nbits)
        w_sort_i = np.argsort(w)

        gs_probs = np.zeros(params.N+1, dtype=float)

        for k in range(len(w)):
            i = w_sort_i[k]
            if k == 0:
                prev_i = i
            else:
                prev_i = w_sort_i[k-1]
            # stop looping if the energy increases from the ground state
            if w[i] > w[prev_i]:
                break
            
            vec = v[:,i]
            for j in range(params.N+1):
                gs_probs[j] += np.abs( np.vdot(vec, statevectors[j]) )**2
        prob_plot.plot(np.arange(params.N+1)*params.db, gs_probs, 'bs-')
        prob_plot.set_ylim([0.0, 1.0])
        prob_plot.grid()

    fig.tight_layout()

    plt.savefig(params.fig_path+params.run_name+run_id+'.png')

def log_data(title, params, E, times, alist):
    np.savetxt(params.log_path+params.run_name+title+'_energy.csv', E, delimiter=',')
    np.savetxt(params.log_path+params.run_name+title+'_iter_time.csv', times, delimiter=',')
    np.save(params.log_path+params.run_name+'_alist.npy', alist, allow_pickle=True)

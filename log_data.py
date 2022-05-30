import numpy as np
import matplotlib.pyplot as plt

import hamiltonians

def plot_data(fig_title, run_id, params, E, statevectors, gs_flag, prob_flag):
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
    if gs_flag:
        w,v = hamiltonians.get_gs(params.hm_list, params.nbits)
        eig_line = energy_plot.axhline(y=w.real, color='k', linestyle='--', label='Ground State Energy')
    
    energy_plot.set_xlabel('Imaginary Time')
    energy_plot.set_ylabel('Energy')
    energy_plot.grid()
    energy_plot.legend(loc='best')

    if prob_flag:
        if not gs_flag:
            w,v = hamiltonians.get_gs(params.hm_list, params.nbits)

        gs_probs = np.zeros(params.N+1, dtype=float)
        for j in range(params.N+1):
            gs_probs[j] += np.abs( np.vdot(v, statevectors[j]) )**2
            
        prob_plot.plot(np.arange(params.N+1)*params.db, gs_probs, 'bs-')
        prob_plot.set_ylim([0.0, 1.0])
        prob_plot.grid()

    fig.tight_layout()

    plt.savefig(params.fig_path+params.run_name+run_id+'.png')

def log_data(title, params, E, times, alist):
    np.savetxt(params.log_path+params.run_name+title+'_energy.csv', E, delimiter=',')
    np.savetxt(params.log_path+params.run_name+title+'_iter_time.csv', times, delimiter=',')
    np.save(params.log_path+params.run_name+title+'_alist.npy', np.asarray(alist,dtype=object), allow_pickle=True)

def plot_all_drifts(params, fig_title, run_names, num_runs, padding):
    mean_Es  = np.zeros((len(run_names), params.N+1))
    std_devs = np.zeros((len(run_names), params.N+1))
    for i in range(len(run_names)):
        run_name = run_names[i]

        if run_name == 'drift_none':
            mean_Es[i]  = np.loadtxt(params.log_path+run_name+'-'+str(0).zfill(padding)+'_energy.csv',delimiter=',')
        else:
            Es = np.zeros((num_runs, params.N+1))
            for run in range(num_runs):
                Es[run] = np.loadtxt(params.log_path+run_name+'-'+str(run).zfill(padding)+'_energy.csv',delimiter=',')
            mean_Es[i]  = np.mean(Es,axis=0)
            std_devs[i] = np.std(Es, axis=0)
    
    fig,axs = plt.subplots(1,1,figsize=(6,5))

    fig.suptitle(fig_title, fontsize=16)
    plt.subplots_adjust(top=0.85)

    styles = ['k.-', 'ro-', 'gs-', 'b^-']

    for i in range(len(run_names)):
        # axs.plot(np.arange(0.0, params.N+1)*params.db, mean_Es[i], styles[i], label=run_names[i])
        axs.errorbar(np.arange(0.0, params.N+1)*params.db, mean_Es[i], yerr=std_devs[i],fmt=styles[i], label=run_names[i])
    
    w,v = hamiltonians.get_gs(params.hm_list, params.nbits)
    axs.axhline(y=w.real, linestyle='--', color='k', label='Ground State Energy')

    axs.set_xlabel('Imaginary Time')
    axs.set_ylabel('Energy')
    axs.grid()
    axs.legend(loc='best')
    fig.tight_layout()
    plt.savefig(params.fig_path+'all_drifts.png')

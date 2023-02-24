import sys
import time
import traceback
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hamiltonians import Hamiltonian
from qnute_params import QNUTE_params as Params
from qnute_output import QNUTE_output as Output
from qnute import qnute

from helpers import get_k_local_domains, int_to_base, base_to_int, exp_mat_psi

def get_random_complex_vector(n):
    '''
    generates a uniformly random complex vector in C^n
    '''
    x = np.random.normal(size=n)
    x /= np.linalg.norm(x)
    phi = np.random.uniform(low=-np.pi, high=np.pi,size=n)
    return np.exp(1j*phi)*x

def get_random_k_local_hamiltonian(k, d, l, qubit_map, domains=None):
    if domains is None:
        domains = get_k_local_domains(k, d, l)
    
    hm_list = [ [ [], np.zeros(3**len(domain),dtype=complex), list(domain) ] for domain in domains ]
    num_coeffs = 0
    for hm in hm_list:
        ndomain = len(hm[2])
        num_coeffs += 3**ndomain
        for p in range(3**ndomain):
            pstring = int_to_base(p, 3, ndomain)
            for i in range(ndomain):
                pstring[i] += 1
            hm[0].append(base_to_int(pstring, 4))
    coeffs = get_random_complex_vector(num_coeffs)
    start = 0
    for hm in hm_list:
        hm[1] = coeffs[start:start+len(hm[1])]
        start += len(hm[1])
    
    return Hamiltonian(hm_list, d, l, qubit_map)

def save_numerical_evolution(H_mat, psi0, dt, N, path:str):
    times = np.arange(0,N+1,1)*dt
    svs = np.zeros((N+1, psi0.shape[0]),dtype=complex)
    for i in range(len(times)):
        svs[i] = exp_mat_psi(H_mat*times[i], psi0)
        svs[i] /= np.linalg.norm(svs[i])
    rdf = pd.DataFrame(np.real(svs))
    idf = pd.DataFrame(np.imag(svs))
    rdf.insert(0,'t',times)
    idf.insert(0,'t',times)
    rdf = rdf.sort_values('t')
    idf = idf.sort_values('t')
    
    if not os.path.exists(path):
        os.makedirs(path)
    rdf.to_csv(path+'num_statevectors_real.csv',index=False)
    idf.to_csv(path+'num_statevectors_imag.csv',index=False)
    return svs

def get_statevectors_from_csv(file):
    rdf = pd.read_csv(file+'_real.csv')
    idf = pd.read_csv(file+'_imag.csv')
    return rdf.drop('t',axis=1).to_numpy() + 1.0j * idf.drop('t',axis=1).to_numpy()

class Run_Params:
    def __init__(self,
    num_expts,
    k,l,d,qubit_map,
    Ds, 
    dts, T,
    num_shots,
    init_sv,
    objective_meas_list=[],
    delta=0.1,
    taylor_norm_flag=False,
    taylor_truncate_h=-1,
    taylor_truncate_a=-1,
    backend=None,
    run_id='',
    log_dir='logs/',
    fig_data_string='figure_data/',
    fig_dir='figs/',
    separator='----------\n'
    ):
        self.num_expts=num_expts
        digits = int(np.floor(np.log10(num_expts))+1)
        if digits < 3:
            digits = 3
        self.digits = digits
        self.k=k
        self.l=l
        self.d=d
        self.domains = get_k_local_domains(k,d,l)
        self.qubit_map=qubit_map
        self.Ds=Ds
        self.dts=dts
        self.T=T
        self.Ns=np.ceil(T/dts).astype(int)
        self.num_shots=num_shots
        self.init_sv=init_sv
        self.objective_meas_list=objective_meas_list
        self.delta=delta
        self.taylor_norm_flag=taylor_norm_flag
        self.taylor_truncate_h=taylor_truncate_h
        self.taylor_truncate_a=taylor_truncate_a
        self.backend=backend
        
        self.run_id = run_id.strip()
        if len(self.run_id) == 0:
            t = time.localtime()
            # current_time = time.strftime('%Y-%m-%d-%H-%M-%S',t)
            self.run_id = time.strftime('%Y-%m-%d',t)
        ham_string = 'd={}/l={}/k={}/'.format(d,l,k)
        self.log_path=log_dir + ham_string + self.run_id + '/'
        self.fig_data_path=self.log_path + fig_data_string
        self.figure_path=fig_dir + ham_string + self.run_id + '/'

        self.separator = separator

def start_log(run_params:Run_Params):
    print('Total number of experiments:', run_params.num_expts)
    print('k={}, lattice_dim={}, lattice_bound={}'.format(run_params.k,run_params.d,run_params.l))
    print('Qubit mapping:')
    if run_params.qubit_map is None:
        print('\tDefault 1-D mapping')
    else:
        for key in run_params.qubit_map.keys():
            print('\t{} -> {},'.format(key, run_params.qubit_map[key]))
    if run_params.objective_meas_list is None or len(run_params.objective_meas_list) == 0:
        print('No objective measurements')
    else:
        print('List of objective measurements:')
        for m_list in run_params.objective_meas_list:
            qbits = m_list[1]
            for p in m_list[0]:
                pstring = int_to_base(p,4,len(qbits))
                m_name = ''
                for i in range(len(qbits)):
                    if pstring[i] == 0: m_name += 'I'
                    else: m_name += chr(ord('X')+pstring[i]-1)
                    m_name += '_'
                    m_name += str(qbits[i])
                    if i < len(qbits) - 1: m_name += ' '
                print('\t{},'.format(m_name))

    print(run_params.separator)

def run_random_hamiltonian_experiments(run_params:Run_Params):
    for i in range(run_params.num_expts):
        print('Started Experiment #{} at {}'.format(format(i+1,'0{}d'.format(run_params.digits)), time.strftime('%I:%M:%S %p, %d %b %Y', time.localtime())))
        expt_path = run_params.log_path + 'expt_{}/'.format(format(i+1,'0{}d'.format(run_params.digits)))
        if not os.path.exists(expt_path):
            os.makedirs(expt_path)

        H = get_random_k_local_hamiltonian(run_params.k, run_params.d, run_params.l, run_params.qubit_map, run_params.domains)
        print('Randomly Generated Effective Hamiltonian:')
        H.print()
        H.to_csv(expt_path+'ham.csv')
        print('Saved Hamiltonian description in \'{}ham.csv\''.format(expt_path))
        print(run_params.separator)

        params = Params(H)
        for t_i in range(len(run_params.dts)):
            dt = run_params.dts[t_i]
            N  = run_params.Ns[t_i]
            t = np.arange(0,N+1,1)*dt
            print('Numerically calculating normalized non-unitary evolution')
            num_svs = save_numerical_evolution(H.get_matrix(), run_params.init_sv, dt, N, expt_path+'/dt={}/'.format(dt))
            print(run_params.separator)
            for D in run_params.Ds:
                params.load_hamiltonian_params(D,False,True)
                for trotter in [False,True]:
                    for shots in run_params.num_shots:
                        print('Setting QNUTE run parameters:')
                        print('\tD={}'.format(D))
                        print('\tdt={}'.format(dt))
                        print('\tdelta={}'.format(run_params.delta))
                        print('\tN={}'.format(N))
                        print('\tnum_shots={}'.format(shots))
                        print('\ttaylor_norm_flag={}'.format(run_params.taylor_norm_flag))
                        print('\ttaylor_truncate_h={}'.format(run_params.taylor_truncate_h))
                        print('\ttaylor_truncate_a={}'.format(run_params.taylor_truncate_a))
                        print('\ttrotter_flag={}'.format(trotter))
                        params.set_run_params(dt,run_params.delta,N,shots,run_params.backend,
                                              init_sv=run_params.init_sv,
                                              taylor_norm_flag=run_params.taylor_norm_flag, 
                                              taylor_truncate_h=run_params.taylor_truncate_h, 
                                              taylor_truncate_a=run_params.taylor_truncate_a, 
                                              trotter_flag=trotter,
                                              objective_meas_list=run_params.objective_meas_list)
                        instance_path = expt_path + 'dt={}/D={}/trotter={}/shots={}/'.format(dt,D,trotter,shots)
                        if not os.path.exists(instance_path):
                            os.makedirs(instance_path)

                        print('Started QNUTE instance at', time.strftime('%I:%M:%S %p, %d %b %Y',time.localtime()))
                        output = qnute(params, log_to_console=True)
                        print('Finished QNUTE instance at', time.strftime('%I:%M:%S %p, %d %b %Y',time.localtime()))

                        output.log_output('run', path=instance_path)
                        print('Saved QNUTE outputs in \'{}\''.format(instance_path))
                        
                        print('Calculating fidelities')
                        fids = np.abs(np.diag( num_svs.conj() @ output.svs.T ))
                        fid_df = pd.DataFrame(data = {'t':t, 'Fidelity': fids})
                        fid_df.to_csv(instance_path+'fidelity_to_taylor.csv',index=False)
                        
                        if shots == 0:
                            s0_svs = get_statevectors_from_csv(instance_path+'run_statevectors')
                        if shots > 0:
                            # Compare to the shots=0 run with the same parameters
                            fids = np.abs(np.diag( s0_svs.conj() @ output.svs.T ))
                            fid_df = pd.DataFrame(data = {'t':t, 'Fidelity': fids})
                            fid_df.to_csv(instance_path+'fidelity_to_0_shots.csv',index=False)
                        print(run_params.separator)

        print('Finished Experiment #{} at {}'.format(format(i+1,'0{}d'.format(run_params.digits)), time.strftime('%I:%M:%S %p, %d %b %Y', time.localtime())))
        print(run_params.separator)

def generate_figure_data(run_params:Run_Params):
    for D in run_params.Ds:
        for t_i in range(len(run_params.dts)):
            dt = run_params.dts[t_i]
            N = run_params.Ns[t_i]
            for trotter in [False,True]:
                for shots in run_params.num_shots:
                    instance_string = 'dt={}/D={}/trotter={}/shots={}/'.format(dt,D,trotter,shots)
                    data_path = run_params.fig_data_path + instance_string
                    if not os.path.exists(data_path):
                        os.makedirs(data_path)
                    for i in range(run_params.num_expts):
                        expt_path = run_params.log_path + 'expt_{}/'.format(format(i+1,'0{}d'.format(run_params.digits)))
                        instance_path = expt_path + instance_string
                        if i == 0:
                            taylor_fid_df = pd.read_csv(instance_path+'fidelity_to_taylor.csv',names=['t','f1'],header=0)
                        else:
                            taylor_fid_df = taylor_fid_df.join(pd.read_csv(instance_path+'fidelity_to_taylor.csv',usecols=['f{}'.format(i+1)],names=['t','f{}'.format(i+1)],header=0))
                    
                    taylor_mean_df = taylor_fid_df.filter(like='f').agg((np.mean, np.std), axis=1, ddof=0)
                    taylor_mean_df.insert(0,'t',taylor_fid_df['t'])
                    taylor_mean_df.to_csv(data_path+'mean_fidelity_to_taylor.csv',index=False)
                    
                    if shots > 0:
                        for i in range(run_params.num_expts):
                            expt_path = run_params.log_path + 'expt_{}/'.format(format(i+1,'0{}d'.format(run_params.digits)))
                            instance_path = expt_path + instance_string
                            if i == 0:
                                s0_fid_df = pd.read_csv(instance_path+'fidelity_to_0_shots.csv',names=['t','f1'],header=0)
                            else:
                                s0_fid_df = s0_fid_df.join(pd.read_csv(instance_path+'fidelity_to_0_shots.csv',usecols=['f{}'.format(i+1)],names=['t','f{}'.format(i+1)],header=0))
                        # Timewise mean and std
                        t_s0_mean_df = s0_fid_df.filter(like='f').agg((np.mean, np.std),axis=1,ddof=0)
                        t_s0_mean_df.insert(0,'t',s0_fid_df['t'])
                        t_s0_mean_df.to_csv(data_path+'timewise_mean_fidelity_to_0_shots.csv',index=False)
                        
                        #Overall mean and std
                        if shots == run_params.num_shots[1]:
                            all_s0_mean_df = pd.DataFrame(s0_fid_df.filter(like='f').stack().agg((np.mean,np.std),ddof=0),columns=[shots])
                        else:
                            all_s0_mean_df.insert(0,shots,s0_fid_df.filter(like='f').stack().agg((np.mean,np.std),ddof=0))
                # Overall mean and std wrt number of shots
                all_s0_mean_df.to_csv(run_params.fig_data_path+ 'dt={}/D={}/trotter={}/'.format(dt,D,trotter)+'overall_mean_fidelity_to_0_shots.csv')

def plot_fidelity_over_time(D, dts, fig_data_path, fig_path, fig_name, dpi=300):
    fig,axs = plt.subplots(2,1,sharex=True,sharey=True)

    for t_i in range(len(dts)):
        dt = dts[t_i]
        # N=run_params.Ns[t_i]
        for trotter in [False, True]:
            instance_path = 'dt={}/D={}/trotter={}/shots={}/'.format(dt,D,trotter,0)
            df = pd.read_csv(fig_data_path+instance_path+'mean_fidelity_to_taylor.csv')
            axs[int(trotter)].errorbar(df['t'],df['mean'],yerr=df['std'], label='dt={}'.format(dt))
    handles,labs = axs[1].get_legend_handles_labels()
    fig.suptitle('QNUTE Fidelity to Normalized Time Evolution')
    axs[0].set_title('Full Update')
    axs[1].set_title('Trotterized Update')
    for ax in axs:
        ax.yaxis.grid(True)
    
    fig.legend(handles,labs, 
               loc='lower center',
               title='Time Step Size',
               shadow=True,
               ncol=len(dts)
              )
    fig.subplots_adjust(hspace=0.4,top=0.8,bottom=0.25)
    
    fig.text(0.5,0.145, 'Time', ha='center')
    fig.text(0.04,0.5, 'Fidelity', va='center',rotation='vertical')
    
    fig.set_facecolor('white')
    plt.savefig(fig_path + fig_name,dpi=dpi)
    
    plt.close()

def plot_fidelity_over_time_shots(D,num_shots, dt, fig_data_path, fig_path, fig_name, dpi=300):
    fig,axs = plt.subplots(2,1,sharex=True,sharey=True)
    
    for shots in num_shots:
        if shots == 0:
            continue
        for trotter in [False, True]:
                instance_path = 'dt={}/D={}/trotter={}/shots={}/'.format(dt,D,trotter,shots)
                df = pd.read_csv(fig_data_path+instance_path+'timewise_mean_fidelity_to_0_shots.csv')
                axs[int(trotter)].errorbar(df['t'],df['mean'],yerr=df['std'], label=str(shots))
    handles,labs = axs[1].get_legend_handles_labels()
    fig.subplots_adjust(hspace=0.4,top=0.8)
    fig.suptitle('Fidelity to QNUTE using Theoretical Expectation')
    axs[0].set_title('Full Update')
    axs[1].set_title('Trotterized Update')
    
    for ax in axs:
        ax.yaxis.grid(True)
        
    fig.legend(handles,labs, 
               loc='lower center',
               title='Number of Measurement Samples',
               shadow=True,
               ncol=len(num_shots)-1
              )
    fig.subplots_adjust(hspace=0.4,top=0.8,bottom=0.25)
    
    fig.text(0.5,0.145, 'Time', ha='center')
    fig.text(0.04,0.5, 'Fidelity', va='center',rotation='vertical')
    
    fig.set_facecolor('white')
    
    plt.savefig(fig_path + fig_name,dpi=dpi)
    plt.close()

def plot_s0_fidelity_bars(dfs, fig_path, fig_name, dpi=300):
    fig,ax = plt.subplots()
    
    width=0.25

    
    for i in range(2):
        df = dfs[i]
        offset = width*i
        y_data = list(df.loc['mean'])[::-1]
        y_error = list(df.loc['std'])[::-1]

        x_labels = list(df.columns)[::-1]
        x_points = np.arange(len(x_labels))
        ax.bar(x_points + offset, y_data, width, yerr=y_error,alpha=0.5,capsize=10, label=['Full Update', 'Trotter Update'][i])
    
    ax.set_xticks(x_points+width/2)
    ax.set_xticklabels(x_labels)

    ax.yaxis.grid(True)
    ax.set_xlabel('Measurement Samples per Observable')
    ax.set_ylabel('Fidelity')
    fig.suptitle('Mean Fidelity to QNUTE using Theoretical Observable Expectations')
    fig.set_facecolor('white')
    fig.subplots_adjust(top=0.9,bottom=0.2)
    fig.legend(loc='lower center', ncol=2,shadow=True)
    plt.savefig(fig_path + fig_name,dpi=dpi)
    plt.close()

def generate_figures(run_params:Run_Params):
    fig_names = ['fidelity_to_0_shots_time/', 'fidelity_to_taylor/', 'fidelity_to_0_shots_bars/']
    for name in fig_names:
        if not os.path.exists(run_params.figure_path+name):
            os.makedirs(run_params.figure_path+name)

    for D in run_params.Ds:
        plot_fidelity_over_time(D,run_params.dts,run_params.fig_data_path, run_params.figure_path + fig_names[1], 'D={}.png'.format(D))
        for t_i in range(len(run_params.dts)):
            dt = run_params.dts[t_i]
            plot_fidelity_over_time_shots(D,run_params.num_shots, dt, run_params.fig_data_path, run_params.figure_path + fig_names[0], 'D={} dt={}.png'.format(D,dt))
            
            dfs = []
            for trotter in [False, True]:
                instance_string = 'dt={}/D={}/trotter={}/'.format(dt,D,trotter)
                dfs.append(pd.read_csv(run_params.fig_data_path + instance_string + 'overall_mean_fidelity_to_0_shots.csv',header=0,index_col=0))
            plot_s0_fidelity_bars(dfs, run_params.figure_path + fig_names[2], 'D={} dt={}.png'.format(D,dt))

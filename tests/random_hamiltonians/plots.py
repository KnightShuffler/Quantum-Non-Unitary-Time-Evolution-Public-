import matplotlib.pyplot as plt
import numpy as np
import h5py
from typing import List

from .experiment_params import Experiment_Params

def plot_expts(file:h5py.File, expt_nos:List[int], figpath='figs/random_hamiltonians'):
    fig, ax = plt.subplots(figsize=(10,6))
    color='green'
    for expt_no in expt_nos:
        stats = file[f'Fidelities/Stats/expt_{expt_no}']
        dt = stats.attrs['dt']
        N = stats.attrs['N']
        time_steps = np.linspace(0.0, N*dt, N+1)
        print(stats.dtype)
        #flag=stats.attrs['trotter_flag']
        ax.plot(time_steps, stats['q2'], label=f'Median (Expt #{expt_no})', color=color, marker='*')
        
        ax.fill_between(time_steps, stats['q1'], stats['q3'], alpha=0.4, label='Quartiles', color=color)
        ax.fill_between(time_steps, stats['q0'], stats['q4'], alpha=0.3, label='Range', color='grey')
        
        color='blue'
    ax.set_xlabel('Time')
    ax.set_ylabel('Average Fidelity')
    ax.set_title(f'Experiment #{expt_nos}')
    ax.legend()
    ax.grid(True)

    top_id = file['Hamiltonians'].attrs['topology_id']

    fig.suptitle(f'Median Fidelity for Topology ID: {top_id}')

    figname = f'{top_id}_expts'
    for n in expt_nos:
        figname += f'_{n}'

    fig.savefig(f'{figpath}/{figname}.png')

from .import loop_expt_params
import logging

def plot_line(ax:plt.axes, expt_params:Experiment_Params, stats):
    time_steps = np.arange(0,expt_params.N+1,1)*expt_params.dt
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][expt_params.D//2]
    logging.info(time_steps.shape)
    logging.info(stats['q2'].shape)
    ax.plot(time_steps, stats['q2'],
            label=f'D={expt_params.D} {"(Trotterized)" if expt_params.trotter_flag else ""}',
            linestyle='-' if expt_params.trotter_flag else '--',
            color=color)
    ax.fill_between(time_steps, stats['q1'], stats['q3'], alpha=0.4,
                    color=color)#, label='Quartiles')
    ax.fill_between(time_steps, stats['q0'], stats['q4'], alpha=0.2,
                    color=color)#, label='Range')
    return stats['q2']
    
def plot_errors(ax:plt.axes, expt_params:Experiment_Params, stats, centers):
    time_steps = np.arange(0,expt_params.N+1,1)*expt_params.dt
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][expt_params.D//2 + 1]
    ax.errorbar(time_steps, centers, yerr=stats['std'],
                color=color, linestyle='', fmt='o:',
                capsize= 4 if expt_params.trotter_flag else 8,
                capthick=1)

def save_fig(dt:float, delta:float,fig:plt.figure, ax:plt.axes, 
             top_id:str,
             figpath:str='figs/random_hamiltonians',
             legend=True):
        figname = f'dt={dt:0.3f}_delta={delta:0.3f}'

        ax.set_ylim(0.75, 1.025)
        ax.set_yticks(np.arange(0.75,1.01, 0.05))

        # handles,labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.15, 0.85))

        if legend:
            ax.legend()

        fig.subplots_adjust(wspace=0.05,hspace=0.05)
        fig.text(0.5, 0.92,
                    f'Average QNUTE Fidelity for for random k-local Hamiltonians\nTopology {top_id} - dt={dt:0.2f}, delta={delta:0.2f}',
                    ha='center', va='center', fontsize=16
                    )

        fig.savefig(f'{figpath}/{top_id}_{figname}.png')



def plot_multiple_expts(file: h5py.File, EXPT_PARAMS:dict, 
                        figpath='figs/random_hamiltonians'):
    fig_dict = {}
    axs_dict = {}
    top_id = file['Hamiltonians'].attrs['topology_id']
    for dt in EXPT_PARAMS['dt']:
        for delta in EXPT_PARAMS['delta']:
            nrows = len(EXPT_PARAMS['k'])
            ncols = len(EXPT_PARAMS['tth'])
            # Make a new figure
            fig_dict[(dt,delta)], axs_dict[(dt,delta)] = \
                plt.subplots(len(EXPT_PARAMS['k']),len(EXPT_PARAMS['tth']),
                             sharex=True,sharey=True,
                             figsize=(ncols * 8,nrows * 5))
            axs = axs_dict[(dt,delta)]
            for i in range(nrows):
                for j in range(ncols):
                    axs[i,j].grid(True)

            # Add the axis labels in each figure
            flag = nrows>1 and  ncols>1
            for (i, k) in enumerate(EXPT_PARAMS['k']):
                (axs[i, 0] if flag else axs[i]).set_ylabel(f'k={k}', rotation=60, size='large', labelpad=15)
            for (j, tth) in enumerate(EXPT_PARAMS['tth']):
                (axs[-1, j] if flag else axs[j]).set_xlabel('Full Taylor Series' if tth < 0 
                                     else f'{tth}-Term Taylor Series', size='large')
    for (expt_no, expt_params) in enumerate(loop_expt_params(EXPT_PARAMS),start=1):
        logging.debug('  Expt No %i, Params: %s', expt_no, expt_params.__repr__())
        fig = fig_dict[(expt_params.dt, expt_params.delta)]
        ax = axs_dict[(expt_params.dt, expt_params.delta)]
        stats = file[f'Fidelities/Stats/expt_{expt_no}']
        logging.info(f'  index of k = {EXPT_PARAMS["k"].index(expt_params.k)}')
        logging.info(f'  index of tth = {EXPT_PARAMS["tth"].index(expt_params.tth)}')
        if expt_params.num_shots == 0:
            centers = plot_line(ax[EXPT_PARAMS['k'].index(expt_params.k),
                        EXPT_PARAMS['tth'].index(expt_params.tth)] if flag
                    else ax[EXPT_PARAMS['k'].index(expt_params.k)], 
                    expt_params, stats)
        else:
            plot_errors(ax[EXPT_PARAMS['k'].index(expt_params.k),
                        EXPT_PARAMS['tth'].index(expt_params.tth)] if flag
                    else ax[EXPT_PARAMS['k'].index(expt_params.k)], 
                    expt_params, stats, centers)

    for dt in EXPT_PARAMS['dt']:
        for delta in EXPT_PARAMS['delta']:
            fig = fig_dict[(dt,delta)]
            ax = axs_dict[(dt,delta)][0,0]
            save_fig(dt,delta,fig,ax,top_id, figpath, True)


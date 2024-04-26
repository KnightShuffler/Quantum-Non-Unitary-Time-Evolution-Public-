import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .simulate_multidim import get_fourier_amplitudes, get_fourier_eigenstates
from .save_experiments import ExperimentData

time_colormap:Colormap = colormaps.get_cmap('coolwarm').reversed()
plt_norm = plt.Normalize(0,1)
plt_markers:str = '.ds^'

plt.rcParams.update({'font.family': 'sans-serif',
                     })

stat_labels:list[str] = [
    r'$F(t)$',
    r'$\ln\ r(t)$',
    r'MSE$(t)$'
]

def get_interpolation_1d(psi:np.ndarray[float], 
                         num_qbits:int,
                         periodic_bc_flag:bool,
                         dense_x:np.ndarray[float],
                         L:float) ->np.ndarray[float]:
    fourier_amplitudes = get_fourier_amplitudes(psi, num_qbits, periodic_bc_flag)
    f_interpolate = np.zeros(dense_x.shape[0], np.float64)

    for i, (state,factor,freqs) in enumerate(get_fourier_eigenstates(num_qbits, periodic_bc_flag)):
        if not periodic_bc_flag:
            f_interpolate += np.sin(freqs[0]*np.pi*dense_x/L) * fourier_amplitudes[i]
        else:
            if i == 0 or i%2 == 1:
                f_interpolate += np.cos(freqs[0]*np.pi*dense_x/L) * fourier_amplitudes[i]
            else:
                f_interpolate += np.sin(freqs[0]*np.pi*dense_x/L) * fourier_amplitudes[i]

    return f_interpolate

def plot_1d_evolution(ax:Axes,
                      f0:np.ndarray[float],
                      num_qbits:int,
                      periodic_bc_flag:bool,
                      solutions:np.ndarray[float],
                      sample_x:np.ndarray[float],
                      dx:float,L:float,
                      times:np.ndarray[float],
                      plot_times:np.ndarray[int],
                      dense_x:np.ndarray[float]|None=None
                      )->None:
    if dense_x is None:
        step = dx/10
        dense_x = np.arange(0,L+step,step)
    for ti in plot_times:
        if ti == 0:
            color = time_colormap(0.0)
            f = get_interpolation_1d(f0, num_qbits, periodic_bc_flag, dense_x, L)
            ax.plot(dense_x, f, linestyle='--', linewidth=0.5, color=color)
            ax.scatter(sample_x, f0, s=10, color=color)
        else:
            t = times[ti-1]/times[-1]
            color = time_colormap(t)
            f = get_interpolation_1d(solutions[ti-1,:], num_qbits, periodic_bc_flag, dense_x, L)
            ax.plot(dense_x, f, linestyle='--', linewidth=0.5, color=color)
            ax.scatter(sample_x, solutions[ti-1,:], s=10, color=color)


def add_time_color_bar(ax:Axes, T:float)->None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt_norm, cmap=time_colormap), cax=cax)
    cb.set_ticks([0, 1])
    cb.set_ticklabels([f'$t=0.0$', f'$t={T:0.1f}$'])

def add_text(ax:Axes, xoffset, yoffset, letter, size=12, **kwargs):
    ax.text(xoffset, yoffset, letter, transform=ax.transAxes,
            size=size, **kwargs)

def generate_evolution_and_stats_figure(expt_data:ExperimentData,
                                        # f0:np.ndarray[float],
                                        # qite_solutions:np.ndarray[float],
                                        # analytical_solution:np.ndarray[float],
                                        # stat_data:np.ndarray[float],
                                        # D_list:np.ndarray[int]|list[int],
                                        # dx:float,L:float,
                                        # times:np.ndarray[float],
                                        # T:float,
                                        # periodic_bc_flag:bool=False,
                                        stat_time_split=20,
                                        plot_times:np.ndarray[int]=None,
                                        figpath:str='figs/',
                                        figname:str='output'
                                        )->None:
    Nd,Nt,Nx = expt_data.qite_sols.shape
    times = np.arange(0,Nt+1)*expt_data.dt
    if not expt_data.periodic_bc_flag:
        sample_x = np.arange(1,Nx+1)*expt_data.dx
    else:
        sample_x = np.arange(0,Nx)*expt_data.dx
    if plot_times is None:
        plot_times = np.arange(0,Nt+1,stat_time_split)
    
    figure = plt.figure(figsize=(12,8))
    evo_fig:Figure = None
    stat_fig:Figure = None
    evo_fig,stat_fig = figure.subfigures(1,2,width_ratios=(3,2))
    evo_axs:np.ndarray[Axes] = evo_fig.subplots(Nd+1,1,sharex=True,sharey=True)
    stat_axs:np.ndarray[Axes] = stat_fig.subplots(3,1,sharex=True)
    time_slice = slice(0,times.shape[0] + stat_time_split,times.shape[0]//stat_time_split)
    letter = 'a'


    for row,ax in enumerate(evo_axs):
        if row < Nd:
            if not expt_data.periodic_bc_flag:
                plot_1d_evolution_zero_bc(ax, expt_data.f0, expt_data.qite_sols[row,:,:], sample_x, expt_data.dx, expt_data.L, times, plot_times)
            else:
                plot_1d_evolution_periodic_bc(ax, expt_data.f0, expt_data.qite_sols[row,:,:], sample_x, expt_data.dx, expt_data.L, times, plot_times)
            add_text(ax, 0.5, 0.05, f'D={expt_data.D_list[row]} Approximation', size=10, horizontalalignment='center')
            ax.set_ylabel(f'$\\psi_{expt_data.D_list[row]}(x,t)$',fontsize=14)
        else:
            if not expt_data.periodic_bc_flag:
                plot_1d_evolution_zero_bc(ax, expt_data.f0, expt_data.analytical_sol, sample_x, expt_data.dx, expt_data.L, times, plot_times)
            else:
                plot_1d_evolution_periodic_bc(ax, expt_data.f0, expt_data.analytical_sol, sample_x, expt_data.dx, expt_data.L, times, plot_times)
            ax.set_xlabel(r'$x$',fontsize=14)
            ax.set_xlim([-expt_data.dx/2, expt_data.L+expt_data.dx/2])
            add_text(ax, 0.5, 0.05, 'Analytical Solution', size=10, horizontalalignment='center')
            ax.set_ylabel(r'$f(x,t)$',fontsize=14)
            

        add_text(ax, -0.1,0.9, f'\\textbf{"{"}({letter}){"}"}')
        letter = chr(ord(letter) + 1)
        
        add_time_color_bar(ax, expt_data.T)
        
        # ax.set_ylim([0.0,1.0])
        

    for row,ax in enumerate(stat_axs):
        add_text(ax, -0.18, 0.9, f"\\textbf{'{'}({letter}){'}'}")
        letter = chr(ord(letter) + 1)
        for Di,D in enumerate(expt_data.D_list):
            l, = ax.plot(times[time_slice], expt_data.stat_data[row,Di,time_slice], marker=plt_markers[Di],label=f'$D={D}$')
            ax.legend(fancybox=False,shadow=True)
        if row == 2:
            ax.set_xlabel(r'$t$',fontsize=14)
        ax.set_ylabel(stat_labels[row],fontsize=14)
        ax.grid(True)

    evo_fig.subplots_adjust(right=0.85,hspace=0.07)
    stat_fig.subplots_adjust(hspace=0.1)

    if figpath[-1] != '/':
        figpath += '/'
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    figure.savefig(figpath+figname+'.svg')
    figure.savefig(figpath+figname+'.png')
    # plt.show()

def generate_stats_figure(expt_data:ExperimentData,
                          stat_time_split=20,
                          plot_times:np.ndarray[int]=None,
                          figpath:str='figs/',
                          figname:str='output')->None:
    Nd,Nt,Nx = expt_data.qite_sols.shape
    times = np.arange(0,Nt+1)*expt_data.dt
    time_slice = slice(0,times.shape[0] + stat_time_split,times.shape[0]//stat_time_split)

    figure,axs = plt.subplots(1, 3, sharex=True, figsize=(12,4))
    
    plot_stat_figures(axs, expt_data, times, time_slice)

    figure.subplots_adjust(wspace=0.4)

    if figpath[-1] != '/':
        figpath += '/'
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    figure.savefig(figpath+figname+'.svg')
    figure.savefig(figpath+figname+'.png')

def plot_stat_figures(stat_axs:np.ndarray[Axes],
                      expt_data:ExperimentData,
                      times:np.ndarray[float],
                      time_slice:slice):
    letter = 'a'
    for col,ax in enumerate(stat_axs):
        add_text(ax, -0.18, 1.0, f"\\textbf{'{'}({letter}){'}'}")
        letter = chr(ord(letter) + 1)
        for Di,D in enumerate(expt_data.D_list):
            l, = ax.plot(times[time_slice], expt_data.stat_data[col,Di,time_slice], marker=plt_markers[Di],label=f'$D={D}$')
            ax.legend(fancybox=False,shadow=True)
        # if col == 2:
            ax.set_xlabel(r'$t$',fontsize=14)
        ax.set_ylabel(stat_labels[col],fontsize=14)
        ax.grid(True)

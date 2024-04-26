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

def plot_2d_state(ax, psi:np.ndarray[float], 
                  num_qbits:np.ndarray[int], 
                  periodic_bc_flags:np.ndarray[bool],
                  L:np.ndarray[float],
                  *,
                  vmin:float|None = None,
                  vmax:float|None = None
                  ) -> None:
    N = 2**num_qbits
    M_slice = [None,None]
    for i in range(2):
        if periodic_bc_flags[i]:
            M_slice[i] = slice(0,N[i])
        else:
            N[i] += 2
            M_slice[i] = slice(1,N[i]-1)

    m = np.zeros((N[1],N[0]),np.float64)
    m[M_slice[1],M_slice[0]] = psi.reshape(2**num_qbits[1],2**num_qbits[0])

    ax.imshow(m[::-1], cmap='coolwarm', interpolation='none', extent = [0.0, L[0], 0.0, L[1]],
              vmin = np.min(psi) if vmin is None else vmin,
              vmax = np.max(psi) if vmax is None else vmax)

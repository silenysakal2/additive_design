import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
from matplotlib import gridspec
from itertools import cycle
import numpy as np
import math
import os
from IPython.display import FileLink
from tqdm.notebook import trange, tqdm


def setup_latex():
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['text.usetex']=True
    #plt.rc('text.latex', preamble=r'\usepackage[bitstream-charter]{mathdesign}')
    plt.rcParams['font.size'] = 8

    # Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman
    plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})

    matplotlib.use('pgf')
    plt.rcParams["pgf.texsystem"] = "xelatex"
    plt.rcParams["pgf.rcfonts"] = False

    preamble = r'''\usepackage[utf8]{inputenc} %unicode support
    \usepackage[czech]{babel}
    \usepackage[T1]{fontenc}
    \DeclareMathAlphabet{\pazocal}{OMS}{zplm}{m}{n}
    \usepackage{calrsfs}
    \usepackage{amsmath}
    \usepackage{bm}
    \usepackage[bitstream-charter]{mathdesign}
    '''
    plt.rc('text.latex', preamble=preamble)
    plt.rcParams["pgf.preamble"] = preamble

import os
from IPython.display import display, FileLink

def plot_progress(ax, ntrials, nrecalc, maxpro_hist, maxpro_opt_hist, T_hist, dir_name=None):
    x_trials = np.linspace(1, ntrials + nrecalc, ntrials + nrecalc)

    l1, = ax.semilogy(x_trials[:ntrials], maxpro_hist[:ntrials], 'k-', label='Criterion history')
    l2, = ax.semilogy(x_trials[:ntrials], maxpro_opt_hist[:ntrials], 'g-', label='Best (optimal)')
    l3, = ax.semilogy(x_trials[ntrials:], maxpro_opt_hist[ntrials:], 'm-', label='Final hungry state')

    ax.set_ylim(0.9 * np.min(maxpro_opt_hist), 1.01 * np.max(maxpro_hist))

    axT = ax.twinx()
    l4, = axT.semilogy(x_trials, T_hist, 'r-', label='Temperature')

    lines = [l1, l2, l3, l4]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='best')

    ax.set_title('Progress of Simulated Annealing combinatorial optimization')

    # Conditional handling for dir_name
    if dir_name is None:
        SAfilename = 'SimulatedAnnealingProgress.pdf'
    else:
        SAfilename = os.path.join(dir_name, 'SimulatedAnnealingProgress.pdf')

    plt.savefig(SAfilename, dpi=300)
    display(FileLink(SAfilename))


def plot_2D_view(nv, ns, x, ax, vars_to_plot=[0, 1], des_no=None):
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)

    # grid
    grid_points = np.linspace(0, 1, 2, endpoint=True)  # left probability bound
    ax.xaxis.set_ticks(grid_points)
    ax.yaxis.set_ticks(grid_points)
    ax.grid(True)

    u, v = vars_to_plot[0], vars_to_plot[1]

    # plot all points as empty circles
    ax.scatter(x[:, u], x[:, v], s=20, facecolors='none', edgecolors='k', alpha=0.4)

    # unique colors for groups of ns points in each tile
    idx = np.random.permutation(np.linspace(0, 1, 9))
    colors = iter(cycle(cm.rainbow(idx)))

    for xadd in [-1, 0, 1]:
        for yadd in [-1, 0, 1]:
            ax.scatter(x[:, u] + xadd, x[:, v] + yadd, s=300/ns, edgecolors=next(colors), facecolors='none')

    # projections onto axes
    ax.scatter(    x[:, u]    , 0 * x[:, v] - 1 , s=50, marker="|", color='k')
    ax.scatter(0 * x[:, u] - 1,     x[:, v]     , s=50, marker="_", color='k')

    # Set the conditional title
    title = f"Design {des_no} view of variables {vars_to_plot}" if des_no is not None else f"Design view of variables {vars_to_plot}"
    ax.set_title(title)

    plt.show()


def plot_2D_view_comparison(nv, ns, x_ini, x_opt, vars_to_plot=[0, 1], dir_name=None):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.1)  # spacing between plots

    titles = ['Initial configuration', 'Optimized configuration']
    for ax, x, title in zip(axs, [x_ini, x_opt], titles):
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')  # equal aspect ratio

        # Grid
        grid_points = np.linspace(0, 1, 2, endpoint=True)
        ax.set_xticks(grid_points)
        ax.set_yticks(grid_points)
        ax.grid(True)

        u, v = vars_to_plot

        # Plot points as empty circles
        ax.scatter(x[:, u], x[:, v], s=20, facecolors='none', edgecolors='k', alpha=0.4)

        # Unique colors for each group
        idx = np.random.permutation(np.linspace(0, 1, 9))
        colors = iter(cycle(cm.rainbow(idx)))

        for xadd in [-1, 0, 1]:
            for yadd in [-1, 0, 1]:
                ax.scatter(
                    x[:, u] + xadd,
                    x[:, v] + yadd,
                    s=300 / ns,
                    edgecolors=next(colors),
                    facecolors='none'
                )

        # Projections onto axes
        ax.scatter(x[:, u], 0 * x[:, v] - 1, s=50, marker="|", color='k')
        ax.scatter(0 * x[:, u] - 1, x[:, v], s=50, marker="_", color='k')


    # Conditional handling for dir_name
    if dir_name is None:
        design_comparison_filename = '2D_view_comparison.pdf'
    else:
        design_comparison_filename = os.path.join(dir_name, '2D_view_comparison.pdf')

    plt.savefig(design_comparison_filename, dpi=300)
    display(FileLink(design_comparison_filename))

    plt.show()


def make_slice_specifiers(ns, nv, active_v=0, slices: list = []):
    if active_v == (nv - 2):
        return [tuple(slices+[slice(None), slice(None)])]
    else:
        output = []
        for i in range(ns):
            output += make_slice_specifiers(ns, nv, active_v+1, slices+[i])
        return output

def plot_histogram(histogram, ns, nv, tsk_name, dir_name):
    slice_specifiers = make_slice_specifiers(ns, nv)

    if (nv % 2 == 0) or (nv < 5):
        rect = False
        plots = math.ceil(math.sqrt(ns ** (nv-2)))
        fig, axs = plt.subplots(nrows=plots, ncols=plots, figsize = (8,8) )
    else:
        rect = True
        plots = math.ceil(math.sqrt(ns ** (nv-3)))
        fig, axs = plt.subplots(nrows=plots, ncols=plots ** 2, figsize = (16, 16 / ns))


    if nv>2:
        axs = axs.flatten()  # Flatten the array to make indexing easier
    else:
        axs = [axs]

    vmin=math.floor( np.min(histogram) )
    vmax=math.ceil ( np.max(histogram) )

    #manually set the limits for grey
    #vmin=0.5
    #vmax=1.5

    for i in trange(ns ** (nv-2)):
        histogram_2d_slice = histogram[slice_specifiers[i]]
        #vmin=np.min(histogram_2d_slice[histogram_2d_slice > 0])
        #vmax=np.max(histogram_2d_slice)


        im = axs[i].imshow(histogram_2d_slice, cmap='Greys', interpolation='nearest', aspect='auto',
                           vmin=vmin, vmax=vmax)
        axs[i].set_xticklabels([])  # Remove the x-axis tick labels
        axs[i].set_yticklabels([])  # Remove the y-axis tick labels
        axs[i].tick_params(axis='both', which='both', length=0)  # Remove the tick marks

        posx = (i // plots) / plots
        posy = (i % plots) / plots
        sizey = 1 / plots
        if rect:
            posx /= plots
            sizex = sizey / plots
        else:
            sizex = sizey
        axs[i].set_position([posx, posy, sizex, sizey])


    histfilename = os.path.join(dir_name, tsk_name + '_hist.pdf')
    plt.savefig(histfilename, dpi=300)
    display(FileLink(histfilename))


    # Colorbar
    fig_colorbar, axs_cb = plt.subplots(nrows=1, ncols=1, figsize = (8,0.4))
    fig_colorbar.colorbar(im, cax=axs_cb, orientation='horizontal')

    plt.subplots_adjust(bottom=0.6, top=0.95, left=0.2, right=0.8)#, wspace=0.05, hspace=0.05)

    histfilename = os.path.join(dir_name, tsk_name + '_hist_colorbar.pdf')
    plt.savefig(histfilename, dpi=300)
    display(FileLink(histfilename))

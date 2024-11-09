import numpy as np
import matplotlib.pyplot as plt

q = 10

for sc in range(1,4):
    print(f'Case {sc}.')

    xc, yc, _, pc = np.loadtxt(f'../data/p_{sc}_xy_cfd.csv', delimiter=',', unpack=True)
    xm, ym, _, pm = np.loadtxt(f'p_{sc}_xy.csv', delimiter=',', unpack=True)
    
    fig, ax = plt.subplots(nrows=2, figsize=(11,6), constrained_layout=True)
    ax[0].scatter(xc[::q], yc[::q], c=-pc[::q], cmap='RdBu', rasterized=True)
    ax[1].scatter(xm[::q], ym[::q], c=-pm[::q], cmap='RdBu', rasterized=True)
    ax[0].set(aspect=True, xlabel='x', ylabel='y', title=f'CFD (p component)')
    ax[1].set(aspect=True, xlabel='x', ylabel='y', title=f'PINN (p component)')

    ax[0].sharex(ax[0])
    fig.suptitle(f'Pressure Field Comparison in XY Plane: Case {sc}')
    plt.savefig(f'xy_p_case_{sc}.png')

    ###

    xc, _, zc, pc = np.loadtxt(f'../data/p_{sc}_xz_cfd.csv', delimiter=',', unpack=True)
    xm, _, zm, pm = np.loadtxt(f'p_{sc}_xz.csv', delimiter=',', unpack=True)
    
    fig, ax = plt.subplots(nrows=2, figsize=(11,6), constrained_layout=True)
    ax[0].scatter(xc[::q], zc[::q], c=-pc[::q], cmap='RdBu', rasterized=True)
    ax[1].scatter(xm[::q], zm[::q], c=-pm[::q], cmap='RdBu', rasterized=True)
    ax[0].set(aspect=True, xlabel='x', ylabel='z', title=f'CFD (p component)')
    ax[1].set(aspect=True, xlabel='x', ylabel='z', title=f'PINN (p component)')

    ax[0].sharex(ax[0])

    fig.suptitle(f'Pressure Field Comparison in XZ Plane: Case {sc}')

    plt.savefig(f'xz_p_case_{sc}.png')



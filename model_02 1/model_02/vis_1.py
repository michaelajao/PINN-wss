import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 600,
})

for sc in range(2,4):
    print(f'Case {sc}.')

    xcy, ycy, _, pcy = np.loadtxt(f'../data/p_{sc}_xy_cfd.csv', delimiter=',', unpack=True)
    xmy, ymy, _, pmy = np.loadtxt(f'p_{sc}_xy.csv', delimiter=',', unpack=True)
    xcz, _, zcz, pcz = np.loadtxt(f'../data/p_{sc}_xz_cfd.csv', delimiter=',', unpack=True)
    xmz, _, zmz, pmz = np.loadtxt(f'p_{sc}_xz.csv', delimiter=',', unpack=True)
    
    fig, ax = plt.subplots(figsize=(9,3), constrained_layout=True)
    ax.scatter(xcy, ycy, c=-pcy, cmap='RdBu', rasterized=True)
    ax.set(aspect=True, ylim=(-0.02,0.06), xlabel='x', ylabel='y', title=f'CFD')
    plt.savefig(f'p_case_{sc}_xy_cfd.png')

    fig, ax = plt.subplots(figsize=(9,3), constrained_layout=True)
    ax.scatter(xmy, ymy, c=-pmy, cmap='RdBu', rasterized=True)
    ax.set(aspect=True, ylim=(-0.02,0.06), xlabel='x', ylabel='y', title=f'Physics-Informed Neural Network')
    plt.savefig(f'p_case_{sc}_xy_pinn.png')

    fig, ax = plt.subplots(figsize=(9,3), constrained_layout=True)
    ax.scatter(xcz, zcz, c=-pcz, cmap='RdBu', rasterized=True)
    ax.set(aspect=True, ylim=(-0.04,0.04), xlabel='x', ylabel='z', title=f'CFD')
    plt.savefig(f'p_case_{sc}_xz_cfd.png')

    fig, ax = plt.subplots(figsize=(9,3), constrained_layout=True)
    ax.scatter(xmz, zmz, c=-pmz, cmap='RdBu', rasterized=True)
    ax.set(aspect=True, ylim=(-0.04,0.04), xlabel='x', ylabel='z', title=f'Physics-Informed Neural Network')
    plt.savefig(f'p_case_{sc}_xz_pinn.png')

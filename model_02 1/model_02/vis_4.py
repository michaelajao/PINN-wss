import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 600,
})

for sc in range(1,4):
    print(f'Case {sc}.')

    xcy, ycy, _, pcy = np.loadtxt(f'../data/p_{sc}_xy_cfd.csv', delimiter=',', unpack=True)
    xmy, ymy, _, pmy = np.loadtxt(f'p_{sc}_xy.csv', delimiter=',', unpack=True)
    xcz, _, zcz, pcz = np.loadtxt(f'../data/p_{sc}_xz_cfd.csv', delimiter=',', unpack=True)
    xmz, _, zmz, pmz = np.loadtxt(f'p_{sc}_xz.csv', delimiter=',', unpack=True)
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,5), constrained_layout=True)
    ax[0,0].scatter(xcy, ycy, c=-pcy, cmap='RdBu', rasterized=True)
    ax[1,0].scatter(xmy, ymy, c=-pmy, cmap='RdBu', rasterized=True)
    ax[0,1].scatter(xcz, zcz, c=-pcz, cmap='RdBu', rasterized=True)
    ax[1,1].scatter(xmz, zmz, c=-pmz, cmap='RdBu', rasterized=True)
    
    ax[0,0].set(aspect=True, ylim=(-0.02,0.06), xlabel='x', ylabel='y', title=f'CFD')
    ax[1,0].set(aspect=True, ylim=(-0.02,0.06), xlabel='x', ylabel='y', title=f'Physics-Informed Neural Network')
    ax[0,1].set(aspect=True, ylim=(-0.04,0.04), xlabel='x', ylabel='z', title=f'CFD')
    ax[1,1].set(aspect=True, ylim=(-0.04,0.04), xlabel='x', ylabel='z', title=f'Physics-Informed Neural Network')

    fig.suptitle(f'Pressure Field Comparison: Case {sc}')
    plt.savefig(f'p_case_{sc}.png')
    # plt.show()

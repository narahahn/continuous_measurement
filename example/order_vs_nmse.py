"""
3D Continuous measurement at discrete elevation angles.

* point source in a free-field
* microphones moving on a sphere at discrete elevation angles
* spatial sampling equivalent to the Gaussian sampling
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import micarray
from sys import path
path.append('../')
from utils import *
from source import *
from sfs.util import sph2cart
dir_save = '/home/nara/Documents/git/research/2018-05_AES_Milan/paper/graphics'
import matplotlib as mpl
mpl.rc('axes', edgecolor='none', axisbelow=True)

dirname = 'data_aes144'
N = 800
order = np.arange(2, 38, 2)
M = len(order)

E = np.zeros(M)
Omega = np.zeros(M)

for i, m in enumerate(order):
    filename = 'msr_N{:04d}_order{:03d}.npz'.format(N, m)
    npzfile = np.load('{}/{}'.format(dirname, filename))
    E[i] = npzfile['E00']
    Omega[i] = npzfile['Omega']

fs = npzfile['fs']
#M_al = npzfile['modal_bandwidth']
M_al = int(np.ceil(np.exp(1)*np.pi*0.8*fs/2*R/c))
Omega_al = np.pi * fs / N / M_al


figname = 'order_vs_nmse'
fig, ax1 = plt.subplots(figsize=(5.5, 4))
ax1.plot([M_al, M_al], [-500, 100], 'r--')
ax1.plot(order, db(E), 'o-')
ax1.set_xlabel(r'Spatial Bandwidth $\mathcal{M}$')
ax1.set_ylabel(r'Mean NMSE $\mathcal{E}$ / dB')
ax1.grid()
ax1.set_xlim(1, 37)
ax1.set_ylim(-130, 6)
ax1.grid(color='lightgray', linestyle='-');
ax1.tick_params(colors='gray')
fig.savefig('{}/{}.pdf'.format(dir_save, figname))

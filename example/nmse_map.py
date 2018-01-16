"""
3D Continuous measurement at discrete elevation angles.

* point source in a free-field
* microphones moving on a sphere at discrete elevation angles
* spatial sampling equivalent to the Gaussian sampling
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal as signal
import micarray
from utils import *
from source import *
from sfs.util import sph2cart
from sys import path
path.append('../')
dir_save = '/home/nara/Documents/git/research/2018-05_AES_Milan/paper/graphics'
mpl.rc('axes', edgecolor='lightgray', axisbelow=True)

dirname = 'data_aes144'
N = 800

order = 36
filename = 'msr_N{:04d}_order{:03d}.npz'.format(N, order)
npzfile = np.load('{}/{}'.format(dirname, filename))

R = npzfile['R']
c = npzfile['c']
fs = npzfile['fs']
M_al = int(np.ceil(np.exp(1)*np.pi*0.8*fs/2*R/c))
Omega_al = np.pi * fs / N / M_al
h0 = npzfile['h0']
h = npzfile['h']
E00 = npzfile['E00']
phi_target = npzfile['phi_target']
theta_target = npzfile['theta_target']
E_matrix = npzfile['E_matrix']
alpha = npzfile['alpha']
beta = npzfile['beta']


figname = 'nmse_map_order{:03d}_E{}'.format(order, -int(db(E00)))

plt.figure(figsize=(6, 3.5))
im = plt.imshow(db(E_matrix), vmax=0,
                extent=[np.rad2deg(phi_target[0]),
                        np.rad2deg(phi_target[-1]),
                        np.rad2deg(theta_target[-1]),
                        np.rad2deg(theta_target[0])],
                cmap='Blues')
plt.plot(np.rad2deg(alpha), np.rad2deg(beta), 'x', color='black')
plt.xticks(np.arange(0, 360+90, 90))
plt.yticks(np.arange(0, 180+90, 90))
plt.xlabel(r'$\phi$ / deg')
plt.ylabel(r'$\theta$ / deg')
plt.ylim(0, 180)
plt.xlim(0, 360)
cbar = plt.colorbar(im, fraction=0.0232, pad=0.04)
ax = cbar.ax
plt.clim(-130, 0)
ax.text(2, -0.05, 'dB', rotation=0, color='gray')
ax.tick_params(colors='gray')
plt.gca().invert_yaxis()
plt.grid(color='lightgray', linestyle='-')
plt.tick_params(colors='gray')
plt.savefig('{}/{}.pdf'.format(dir_save, figname))

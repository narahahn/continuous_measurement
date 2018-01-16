"""
Plot IRs and TFs of continuous measurement on a sphere at discrete elevations

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

order = 8, 16, 24

q_theta = 33
q_phi = 47

irs = np.zeros((len(order), N))
for ii, m in enumerate((order)):
    filename = 'msr_N{:04d}_order{:03d}.npz'.format(N, m)
    npzfile = np.load('{}/{}'.format(dirname, filename))

    c = npzfile['c']
    fs = npzfile['fs']
    M_al = int(np.ceil(np.exp(1)*np.pi*0.8*fs/2*R/c))
    h0 = npzfile['h0']
    h = npzfile['h']
    phi_target = npzfile['phi_target']
    theta_target = npzfile['theta_target']
    Q_phi = npzfile['Q_phi']
    Q_theta = npzfile['Q_theta']

    h_2d = np.reshape(h, (Q_theta, Q_phi, N))
    irs[ii, :] = h_2d[q_theta, q_phi, :]

h0_2d = np.reshape(h0, (Q_theta, Q_phi, N))
theta0 = np.rad2deg(theta_target[q_theta])
phi0 = np.rad2deg(phi_target[q_phi])

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#17becf',    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22']

figname = 'irs'
plt.figure(figsize=(6, 3.5))
time_ms = np.arange(N)/fs*1000
plt.plot(time_ms, db(h0_2d[q_theta, q_phi, :]),
         lw=4, color='lightgray', label='original')
for ii, m in enumerate(order):
    plt.plot(time_ms, db(irs[ii, :]),
             label='$\mathcal{{M}}$ = {}'.format(m), color=colors[ii+3])
plt.xlabel('$t$ / ms')
plt.ylabel('Amplitude / dB')
plt.legend(loc='lower right')
plt.grid(color='lightgray', linestyle='-')
plt.ylim(-260, -30)
plt.savefig('{}/{}.pdf'.format(dir_save, figname))


figname = 'tfs'
plt.figure(figsize=(6, 3.5))
Nf = N // 2 + 1
freq = np.linspace(0, fs/2, num=Nf, endpoint=True) / 1
plt.plot(freq, db(np.fft.rfft(h0_2d[q_theta, q_phi, :])),
         lw=4, color='lightgray', label='original', linestyle='-')
for ii, m in enumerate(order):
    plt.plot(freq, db(np.fft.rfft(irs[ii, :])),
             label='$\mathcal{{M}}$ = {}'.format(m), color=colors[ii+3])
plt.xlabel(r'$f$ / Hz')
plt.ylabel('Magnitude / dB')
plt.xscale('log')
plt.grid(color='lightgray', linestyle='-')
plt.legend(loc='lower left')
plt.xlim(20, fs/2)
plt.ylim(-57, -27)
plt.savefig('{}/{}.pdf'.format(dir_save, figname))

"""
    Continuous measurement of room impulse responses using a moving microphone
    * point source in a free-field
    * omnidirectional microphone moving on a circle at a constant speed
    * captured signal computed by using fractional delay filters
    * system identification based on spatial interpolation of a given order
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from sys import path
path.append('../')
from utils import *
from source import *

# Constants
c = 343
fs = 44100

# Source
xs = [0, 2, 0]  # Point source
source_type = 'point'

# Receiver
R = 0.5

# Experimental parameters
N = 4410  # FIR filter length
K = 720  # number of target angles
Lf = 24  # fractional delay filter length

# The impulse responses at selected angles
phi_k = np.linspace(0, 2 * np.pi, num=K, endpoint=False)
x_k = [R*np.cos(phi_k), R*np.sin(phi_k), np.zeros_like(phi_k)]
waveform_k, shift_k, offset_k = impulse_response(xs, x_k, 'point', fs)
h0, _, _ = construct_ir_matrix(waveform_k, shift_k, N)


# Plots
nn = np.random.randint(0, K+1)
tau = np.arange(N) / fs * 1000
Nf = int(np.ceil(N/2+1))
freq = np.arange(Nf) / Nf * fs/2
cht_order = np.arange(-K/2, K/2)

plt.figure()
plt.plot(tau, h0[nn,:])
plt.xlim(0, N/fs*1000)
plt.xlabel('$t$ / ms')
plt.title('Impulse Response ($\phi={}^\circ$)'.format(360*nn/K))

plt.figure()
plt.plot(tau, db(h0[nn,:]))
plt.xlim(0, N/fs*1000)
plt.xlabel('$t$ / ms')
plt.ylabel('Amplitude / dB')
plt.title('Impulse Response ($\phi={}^\circ$)'.format(360*nn/K))

plt.figure()
plt.plot(freq, db(np.fft.rfft(h0[nn,:])))
plt.xscale('log')
plt.xlim(0, fs/2)
plt.ylim(-60, 0)
plt.title('Transfer Function ($\phi={}^\circ$)'.format(360*nn/K))

plt.figure()
plt.pcolormesh(np.rad2deg(phi_k), tau, h0.T, cmap='coolwarm')
plt.axis('normal')
plt.colorbar()
plt.clim(-0.05, 0.05)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$\tau$ / ms')
plt.xlim(0, 360)
plt.ylim(0, 20)
plt.title('Impulse Responses')

plt.figure()
plt.pcolormesh(np.rad2deg(phi_k), tau, db(h0).T)
plt.axis('normal')
plt.colorbar(label='dB')
plt.clim(-120, -20)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$\tau$ / ms')
plt.xlim(0, 360)
plt.ylim(0, N/fs*1000)
plt.title('Impulse Responses')

plt.figure()
plt.pcolormesh(np.rad2deg(phi_k), freq/1000, db(np.fft.rfft(h0, axis=-1)).T)
plt.axis('normal')
plt.colorbar(label='dB')
plt.clim(-100, 0)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$f$ / kHz')
plt.xlim(0, 360)
plt.ylim(0, fs/2/1000)
plt.title('Frequency Responses')

plt.figure(figsize=(8, 4))
plt.pcolormesh(cht_order, freq/1000, db(np.fft.fftshift(np.fft.fft2(h0)[:, :Nf], axes=0)).T, vmin=-120)
plt.axis('normal')
plt.colorbar(label='dB')
#plt.clim(-100, 20)
plt.xlabel('CHT order')
plt.ylabel('$f$ / kHz')
plt.xlim(-K/2, K/2)
plt.ylim(0, fs/2/1000)
plt.title('CHT Spectrum')




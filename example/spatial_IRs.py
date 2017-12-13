"""
Simulate anechoic impulse responses.
    
* point source in a free-field
* circular microphone array
* fractional delay filters with oversampling
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
fs = 16000

# Source
xs = [0, 20, 0]  # Point source
source_type = 'point'

# Receiver
R = 0.5

# Experimental parameters
N = 1600  # FIR filter length
K = 360  # number of target angles
# Lf = 24  # fractional delay filter length

# The impulse responses at selected angles
phi = np.linspace(0, 2 * np.pi, num=K, endpoint=False)
x = [R*np.cos(phi), R*np.sin(phi), np.zeros_like(phi)]
#waveform, shift, offset = impulse_response(xs, x, 'point', fs)
waveform, shift, offset = impulse_response([0, -1, 0], x, 'plane', fs)
h, _, _ = construct_ir_matrix(waveform, shift, N)


# Plots
nn = np.random.randint(0, K+1)
tau = np.arange(N) / fs * 1000
Nf = int(np.ceil(N/2+1))
freq = np.arange(Nf) / Nf * fs/2
cht_order = np.arange(-K/2, K/2)

# Fig. Impulse response (randomly selected)
plt.figure()
plt.plot(tau, h[nn, :])
plt.xlim(0, N/fs*1000)
plt.xlabel('$t$ / ms')
plt.title('Impulse Response ($\phi={}^\circ$)'.format(360*nn/K))

# Fig. Impulse response in dB (randomly selected)
plt.figure()
plt.plot(tau, db(h[nn, :]))
plt.xlim(0, N/fs*1000)
plt.xlabel('$t$ / ms')
plt.ylabel('Amplitude / dB')
plt.title('Impulse Response ($\phi={}^\circ$)'.format(360*nn/K))

# Fig. Transfer function (randomly selected)
plt.figure()
plt.plot(freq, db(np.fft.rfft(h[nn, :])))
plt.xscale('log')
plt.xlim(20, fs/2)
plt.ylim(-60, 0)
plt.title('Transfer Function ($\phi={}^\circ$)'.format(360*nn/K))

# Fig. 2D plot of the impulse response coefficients
plt.figure()
plt.pcolormesh(np.rad2deg(phi), tau, h.T, cmap='coolwarm')
plt.axis('normal')
plt.colorbar()
plt.clim(-0.05, 0.05)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$\tau$ / ms')
plt.xlim(0, 360)
#plt.ylim(0, 20)
plt.title('Impulse Responses')
#plt.plot([0, 360], [1.5/c*1000, 1.5/c*1000])
#plt.plot([0, 360], [2.5/c*1000, 2.5/c*1000])
#plt.ylim(4, 8)

# Fig. 2D plot of the impulse response coefficients in dB
plt.figure()
plt.pcolormesh(np.rad2deg(phi), tau, db(h).T)
plt.axis('normal')
plt.colorbar(label='dB')
plt.clim(-120, -20)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$\tau$ / ms')
plt.xlim(0, 360)
#plt.ylim(0, 20)
plt.title('Impulse Responses')

# Fig. 2D plot of the transfer functions
plt.figure()
plt.pcolormesh(np.rad2deg(phi), freq/1000, db(np.fft.rfft(h, axis=-1)).T)
plt.axis('normal')
plt.colorbar(label='dB')
plt.clim(-100, 0)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$f$ / kHz')
plt.xlim(0, 360)
plt.ylim(0, fs/2/1000)
plt.title('Frequency Responses')

# Fig. Circular harmonics spectrum in dB
plt.figure(figsize=(8, 4))
plt.pcolormesh(cht_order, freq/1000, db(np.fft.fftshift(np.fft.fft2(h)[:, :Nf], axes=0)).T, vmin=-120)
plt.axis('normal')
plt.colorbar(label='dB')
plt.xlabel('CHT order')
plt.ylabel('$f$ / kHz')
plt.xlim(-K/2, K/2)
plt.ylim(0, fs/2/1000)
plt.title('CHT Spectrum')

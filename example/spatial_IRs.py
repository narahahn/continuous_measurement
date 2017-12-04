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

# Constants
c = 343
fs = 44100

# Source
xs = [0, 2]  # Point source
source_type = 'point'

# Receiver
R = 0.5

# Experimental parameters
N = 4410  # FIR filter length
K = 720  # number of target angles
Lf = 25  # fractional delay filter length

# oversampling 
Q = 2  # oversampling factor
fpass = fs/2 * 0.8
fstop = fs/2 * 1.0
att = -130
order = Q * 40
#f = fir_linph_ls(fpass, fstop, att, order, Q*fs, density=40)
f = sig.remez(2*order+1, [0, fpass, fstop, Q*fs/2], [1, 10**((att)/20)], weight=[1, 10e5], fs=Q*fs)
if Q != 1:
    w, F = signal.freqz(f)
    plt.figure()
    plt.plot(w/2/np.pi*fs*Q, db(F))
    plt.ylim(bottom=att-10)
        
    plt.figure()
    plt.plot(f)

# The desired impulse responses at selected angles
phi_k = np.linspace(0, 2 * np.pi, num=K, endpoint=False)
distance_k = np.sqrt((R*np.cos(phi_k)-xs[0])**2 + (R*np.sin(phi_k)-xs[1])**2)
delay_k = distance_k / c
weight_k = 1/4/np.pi/distance_k
waveform_k, shift_k, offset_k = fractional_delay(delay_k, Lf, fs=Q*fs, type='lagrange')
if Q != 1:
    hup, _, _ = construct_ir_matrix(waveform_k*weight_k[:, np.newaxis], shift_k, Q*N)
    h0 = signal.resample_poly(hup, 1, Q, axis=-1, window=f) * Q
else:
    h0, _, _ = construct_ir_matrix(waveform_k*weight_k[:, np.newaxis], shift_k, Q*N)

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




"""
Continuous measurement of room impulse responses using a moving microphone.

* point source in a free-field
* omnidirectional microphone moving on a circle at a constant speed
* captured signal computed by using fractional delay filters + oversampling
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
fs = 16000

# Source
xs = [0, 2, 0]  # Point source
source_type = 'point'

# Receiver
R = 0.5
Omega = 2 * np.pi / 16
L = int(2 * np.pi / Omega * fs)
t = (1/fs) * np.arange(L)
phi0 = -1e0
phi = Omega * t + phi0
xm = [R*np.cos(phi), R*np.sin(phi), np.zeros_like(phi)]

# Excitation
N = 800  # excitation period
p = perfect_sequence_randomphase(N)
#p = perfect_sweep(N)

# Experimental parameters
K = 720  # number of target angles
#Lf = 21  # fractional delay filter length
int_order = 7  # spatial interpolation order
Omega_al = c / N / R  # anti-aliasing angular speed

# Captured signal
waveform_l, shift_l, offset_l = impulse_response(xs, xm, 'point', fs)
s = captured_signal(waveform_l, shift_l, p)

# Desired impulse responses at selected angles
phi_k = np.linspace(0, 2 * np.pi, num=K, endpoint=False)
x_k = [R*np.cos(phi_k), R*np.sin(phi_k), np.zeros_like(phi_k)]
delay_k, weight_k = greens_point(xs, x_k)
waveform_k, shift_k, offset_k = impulse_response(xs, x_k, 'point', fs)
h0, _, _ = construct_ir_matrix(waveform_k, shift_k, N)

# System identification
hhat = np.zeros((K, N))
yhat = np.zeros((K, N))
dphi = 2 * np.pi / L * N
#n_phi = phi / dphi
n_phik = (phi_k - phi0) / dphi
L_int = int_order + 1

# Spatial interpolation -- Barycentric form
ii = np.arange(L_int)
w = comb(L_int-1, ii) * (-1)**ii
for n in range(N):
    if L_int % 2 == 0:
        n0 = np.ceil(n_phik - n/N).astype(int)
        Lh = int(L_int/2)
    elif L_int % 2 == 1:
        n0 = np.round(n_phik - n/N).astype(int)
        Lh = int((L_int+1)/2)
    idx_matrix = n0[:, np.newaxis] + (np.arange(-Lh, -Lh+L_int))[np.newaxis, :]
    offset = n0
    shift = n0 - Lh
    waveform = w[np.newaxis, :] / (n_phik[:, np.newaxis] - n/N - idx_matrix)
    waveform /= np.sum(waveform, axis=-1)[:, np.newaxis]
    s_n = s[n::N]
    for k in range(K):
        idx = np.arange(shift[k], shift[k]+L_int).astype(int)
        yhat[k, n] = np.dot(s_n[np.mod(idx, int(L/N))], waveform[k, :])

for k in range(K):
    hhat[k, :] = cxcorr(yhat[k, :], p)


# System identification
#hhat = np.zeros((K, N))
#for k in range(K):
#    y = np.zeros(N)
#    for n in range(N):
#        phitemp = np.mod(phi[n::N], 2*np.pi)
#        sm = s[n::N]
#        idx_sort = np.argsort(phitemp)
#        phitemp = phitemp[idx_sort]
#        sm = sm[idx_sort]
#
#        phitemp = np.concatenate([phitemp-2*np.pi, phitemp, phitemp+2*np.pi])
#        sm = np.concatenate([sm, sm, sm])
#
#        y[n] = fdfilter(phitemp, sm, phi_k[k], order=int_order)
#    hhat[k, :] = cxcorr(y, p)


# Plots
nn = np.random.randint(0, K+1)  # random choice
tau = np.arange(N) / fs * 1000
Nf = int(np.ceil(N/2+1))
freq = np.arange(Nf) / Nf * fs/2
order = np.arange((-K)//2, K//2)

# Fig. Impulse response (for a randomly selected randomly selected)
plt.figure()
plt.plot(tau, h0[nn,:], label='original', linewidth=5, color='lightgray')
plt.plot(tau, hhat[nn,:], label='estimate')
plt.legend(loc='best')
plt.xlabel(r'$t$ / ms')
plt.title('Impulse Response ($\phi={}^\circ$)'.format(360*nn/K))

# Fig. Impulse response in dB (for a randomly selected angle)
plt.figure()
plt.plot(tau, db(h0[nn,:]), label='original', linewidth=5, color='lightgray')
plt.plot(tau, db(hhat[nn,:]), label='estimate')
plt.xlabel(r'$t$ / ms')
plt.ylabel('Amplitude / dB')
plt.legend(loc='best')
plt.title('Impulse Response ($\phi={}^\circ$)'.format(360*nn/K))

# Fig. Transfer function (for a randomly selected angle)
plt.figure()
plt.plot(freq, db(np.fft.rfft(h0[nn,:])), label='original', linewidth=5, color='lightgray')
plt.plot(freq, db(np.fft.rfft(hhat[nn,:])), label='estimate')
plt.legend(loc='best')
plt.xscale('log')
plt.xlim(0, fs/2)
plt.ylim(-60, 0)
plt.xlabel(r'$f$ / Hz')
plt.ylabel('Magnitude / dB')
plt.title('Transfer Function ($\phi={}^\circ$)'.format(360*nn/K))

# Fig. Impulse response coefficients in dB
plt.figure()
plt.pcolormesh(np.rad2deg(phi_k), tau, db(hhat).T)
plt.axis('normal')
cb = plt.colorbar(label='dB')
plt.clim(-100, 0)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$\tau$ / ms')
plt.xlim(0, 360)
plt.ylim(0, N/fs*1000)
plt.title('FIR Coefficient Error')

# Fig. Impulse response coefficient errors in dB
plt.figure()
plt.pcolormesh(np.rad2deg(phi_k), tau, db(h0-hhat).T)
plt.axis('normal')
cb = plt.colorbar(label='dB')
plt.clim(-100, 0)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$\tau$ / ms')
plt.xlim(0, 360)
plt.ylim(0, N/fs*1000)
plt.title('FIR Coefficient Error')

# Fig. Spectral distortion in dB
plt.figure()
plt.pcolormesh(np.rad2deg(phi_k), freq/1000, db(np.fft.rfft(h0-hhat, axis=-1)).T)
plt.axis('normal')
plt.colorbar(label='dB')
plt.clim(-100, 0)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$f$ / kHz')
plt.xlim(0, 360)
plt.ylim(0, fs/2/1000)
plt.title('Spectral Distortion')

# Fig. Normalized system distance in dB
plt.figure()
plt.plot(np.rad2deg(phi_k), db(np.sqrt(np.sum((h0-hhat)**2, axis=-1) / np.sum(h0**2, axis=-1))))
plt.xlim(0, 360)
plt.ylim(-150, 0)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel('NMSE / dB')
plt.title('Normalized Mean Square Error')

# Fig. Desired CHT spectrum
plt.figure(figsize=(10, 4))
plt.pcolormesh(order, freq/1000, db(np.fft.fftshift(np.fft.fft2(h0), axes=0)[:, :Nf]).T, vmin=-120)
plt.axis('normal')
plt.xlabel('CHT order')
plt.ylabel(r'$f$ / kHz')
plt.colorbar()

# Fig. CHT spectrum of the impulse responses
plt.figure(figsize=(10, 4))
plt.pcolormesh(order, freq/1000, db(np.fft.fftshift(np.fft.fft2(hhat), axes=0)[:, :Nf]).T, vmin=-120)
plt.axis('normal')
plt.xlabel('CHT order')
plt.ylabel(r'$f$ / kHz')
plt.colorbar()
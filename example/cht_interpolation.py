"""
Continuous measurement based on modal analysis

* point source in a free-field
* omnidirectional microphone moving on a circle at a constant speed
* captured signal computed by using fractional delay filters + oversampling
* system identification based on modal analysis of each set of N sub-samples
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import micarray
from sys import path
path.append('../')
from utils import *
from source import *

# Constants
c = 343
fs = 8000

# Source
xs = [0, 8, 0]
source_type = 'point'

# Receiver
R = 0.5
Omega = 2 * np.pi / 7
L = int(2 * np.pi / Omega * fs)
t = (1/fs) * np.arange(L)
phi0 = 0  # initial angle
phi = Omega * t + phi0
xm = [R*np.cos(phi), R*np.sin(phi), np.zeros_like(phi)]

# Excitation
N = 800  # excitation period
p = perfect_sequence_randomphase(N)

# Experimental parameters
Omega_al = c / N / R  # anti-aliasing angular speed

# Captured signal and additive noise
rm = np.array(xs)[:, np.newaxis] - np.array(xm)
waveform_l, shift_l, offset_l = impulse_response(xs, xm, source_type, fs)
s = captured_signal(waveform_l, shift_l, p)
snr = -60
s += additive_noise(s, snr)

# Target positions
K = 360  # number of impulse responses
phi_k = np.linspace(0, 2*np.pi, num=K, endpoint=False)  # target angles

# Modal analysis
max_cht_order = (int(L/N)-1) // 2  # maximum modal order
Nf = N // 2 + 1  # number of frequency bins
freq = np.linspace(0, fs/2, num=Nf, endpoint=True)
k = 2 * np.pi * freq / c
Pinv = np.fft.rfft(np.roll(p[::-1], 1))
H_cht = np.zeros((Nf, 2*max_cht_order+1)).astype(complex)
for i in range(N):
    phi_i = phi[i::N]
    s_i = s[i::N]
    P_i = Pinv * np.exp(-1j*2*np.pi/N*np.arange(Nf)*i)
    H_i = P_i[:, np.newaxis] * s_i[np.newaxis, :]
    Psi_i = micarray.modal.angular.cht_matrix(max_cht_order, phi_i, 1/(L/N))
    H_cht += np.matmul(H_i, np.conj(Psi_i.T))

# Compute impulse responses using inverse CHT
Psi = micarray.modal.angular.cht_matrix(max_cht_order, phi_k)  # CHT matrix
H = np.matmul(H_cht, Psi).T  # inverse CHT
h = np.fft.irfft(H, axis=-1) # inverse DFT

# Original impulse responses
x_k = [R*np.cos(phi_k), R*np.sin(phi_k), np.zeros_like(phi_k)]
r_k = np.array(xs)[:, np.newaxis] - np.array(x_k)
waveform_k, shift_k, offset_k = impulse_response(xs, x_k, source_type, fs)
h0, _, _ = construct_ir_matrix(waveform_k, shift_k, N)
H0 = np.fft.rfft(h0, axis=-1)

# Evaluation
system_distance = nmse(h, h0, normalize='each')

# Plots
nn = np.random.randint(0, K)
time_ms = np.arange(0, N) / fs * 1000
freq_kHz = freq / 1000
order = np.arange(-max_cht_order, max_cht_order+1)
phi_target_deg = np.rad2deg(phi_k)

# Fig. Origianal impulse responses
plt.figure()
plt.pcolormesh(phi_target_deg, time_ms, db(h.T), vmin=-100)
plt.colorbar(label='dB')
plt.xlabel(r'$\phi$ / deg')
plt.ylabel(r'$t$ / ms')
plt.title('Desired Impulse Responses')

# Fig. Measured impulse responses
plt.figure()
plt.pcolormesh(phi_target_deg, time_ms, db(h0).T, vmin=-100)
plt.colorbar(label='dB')
plt.xlabel(r'$\phi$ / deg')
plt.ylabel(r'$t$ / ms')
plt.title('Measured Impulse Responses')

# Fig. Filter coefficient errors
plt.figure()
plt.pcolormesh(phi_target_deg, time_ms, db(h0-h).T)
plt.colorbar(label='dB')
plt.xlabel(r'$\phi$ / deg')
plt.ylabel(r'$t$ / ms')
plt.title('Filter Coefficient Errors')

# Fig. Compare impulse responses (randomly selected)
plt.figure()
plt.plot(time_ms, db(h0[nn, :]), label='original')
plt.plot(time_ms, db(h[nn, :]), label='measured')
plt.xlabel(r'$t$ / ms')
plt.ylabel('Amplitude / dB')
plt.legend(loc='best')
plt.title('Comparison of impulse resposnes')

# Fig. Circular harmonics spectrum
plt.figure()
plt.pcolormesh(order, freq_kHz, db(np.fft.fftshift(H_cht, axes=-1)), vmin=-120, cmap='Blues')
plt.colorbar(label='dB')
plt.xlabel('CHT order')
plt.ylabel('$f$ / kHz')
plt.title('CHT Spectrum ($M_{{max}}={}$)'.format(max_cht_order))

# Fig. Spectral distortion
plt.figure()
plt.pcolormesh(phi_target_deg, freq_kHz, db(H-H0).T)
plt.colorbar(label='dB')
plt.xlabel(r'$\phi$ / deg')
plt.ylabel('$f$ / kHz')
plt.title('Spectral Distortion')

# Fig. System distance
plt.figure()
plt.plot(phi_target_deg, db(system_distance))
plt.xlabel(r'$\phi$ / deg')
plt.ylabel('System Distance / dB')
plt.ylim(-120, 0)
plt.title('System Distance')

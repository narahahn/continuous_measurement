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

# Constants
c = 343
fs = 8000

# Source
xs = [0, 8, 0]
source_type = 'point'

# Excitation
N = 400  # excitation period
p = perfect_sequence_randomphase(N)

# Sampling scheme
max_sht_order = 20
x, weights = np.polynomial.legendre.leggauss(max_sht_order+1)
weights *= np.pi / (max_sht_order+1)
theta = np.arccos(x)
channels = max_sht_order + 1
Q = len(theta)
M = 2*max_sht_order + 2

# Microphones
R = 0.2
Omega = 2*np.pi*fs / (N*M)
L = int(2*np.pi/Omega*fs)
t = (1/fs) * np.arange(L)
phi0 = 0
phi = Omega*t + phi0
positions = np.stack((
        R*np.sin(theta[:, np.newaxis])*np.cos(phi), \
        R*np.sin(theta[:, np.newaxis])*np.sin(phi), \
        R*np.cos(theta[:, np.newaxis])*np.ones_like(phi)), axis=-1)

# Experimental parameters
#Omega_al = c / N / R  # anti-aliasing angular speed


# Captured signal and additive noise
snr = -np.infty
s = np.zeros((Q, L))
for q in range(Q):
    xm = np.squeeze(positions[q, :, :])
    waveform_l, shift_l, offset_l = impulse_response(xs, xm, source_type, fs)
    s[q, :] = captured_signal(waveform_l, shift_l, p)
    s[q, :] += additive_noise(s[q, :], snr)

# Target positions
_, _, weights = micarray.modal.angular.grid_gauss(max_sht_order)
azi = np.linspace(0, 2*np.pi, 2*max_sht_order+2, endpoint=False)
x, weights = np.polynomial.legendre.leggauss(max_sht_order+1)
elev = np.arccos(x)


h = np.zeros((Q, M, N))
max_cht_order = (int(L/N)-1) // 2
Nf = N // 2 + 1
freq = np.linspace(0, fs/2, num=Nf, endpoint=True)
k = 2 * np.pi * freq / c
Pinv = np.fft.rfft(np.roll(p[::-1], 1))
for q in range(Q):
    H_cht = np.zeros((Nf, 2*max_cht_order+1)).astype(complex)
    for i in range(N):
        phi_i = phi[i::N]
        s_i = s[q, i::N]
        P_i = Pinv * np.exp(-1j*2*np.pi/N*np.arange(Nf)*i)
        H_i = P_i[:, np.newaxis] * s_i[np.newaxis, :]
        Psi_i = micarray.modal.angular.cht_matrix(max_cht_order, phi_i, 1/M)
        H_cht += np.matmul(H_i, np.conj(Psi_i.T))
    Psi = micarray.modal.angular.cht_matrix(max_cht_order, azi)
    H = np.matmul(H_cht, Psi).T
    h[q, :, :] = np.fft.irfft(H, axis=-1)
hrs = np.reshape(h, (Q*M, N))

# Original impulse responses
azi = np.linspace(0, 2*np.pi, 2*max_sht_order+2, endpoint=False)
x, weights = np.polynomial.legendre.leggauss(max_sht_order+1)
elev = np.arccos(x)
h0 = np.zeros_like(h)
for i, theta in enumerate(elev):
    x_k = [R*np.sin(theta)*np.cos(azi), R*np.sin(theta)*np.sin(azi), R*np.cos(theta)*np.ones_like(azi)]
    waveform_k, shift_k, offset_k = impulse_response(xs, x_k, source_type, fs)
    htemp, _, _ = construct_ir_matrix(waveform_k, shift_k, N)
#    H0 = np.fft.rfft(h0, axis=-1)
    h0[i, :, :] = htemp
h0rs = np.reshape(h0, (Q*M, N))
    
# Evaluation
system_distance = nmse(hrs, h0rs, normalize='each')

# Plots
nn = np.random.randint(0, Q*M)
plt.figure()
plt.plot(db(hrs[nn, :]))
plt.plot(db(h0rs[nn, :]))

plt.figure()
plt.plot(db(system_distance))

plt.figure()
plt.pcolormesh(db(hrs-h0rs))
plt.colorbar()

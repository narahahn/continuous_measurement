"""
Plane wave decomposition using continuously measured impulse responses

* point source in a free-field
* omnidirectional microphone moving on a circle at a constant speed
* captured signal computed by using fractional delay filters + oversampling
* system identification based on spatial interpolation of a given order
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
fs = 16000

# Source
xs = [0, 8, 0]  # Point source
source_type = 'point'

# Receiver
R = 0.5
Omega = 2 * np.pi / 20
L = int(2 * np.pi / Omega * fs)
t = (1/fs) * np.arange(L)
phi0 = -1e0
phi = Omega * t + phi0
xm = [R*np.cos(phi), R*np.sin(phi), np.zeros_like(phi)]

# Excitation
N = 1600  # excitation period
p = perfect_sequence_randomphase(N)
#p = perfect_sweep(N)

# Experimental parameters
K = 360  # number of target angles
#Lf = 21  # fractional delay filter length
int_order = 5  # spatial interpolation order
Omega_al = c / N / R  # anti-aliasing angular speed

# Captured signal
waveform_l, shift_l, offset_l = impulse_response(xs, xm, 'point', fs)
s = captured_signal(waveform_l, shift_l, p)
additive_noise = np.random.randn(len(s))
Es = np.std(s)
En = np.std(additive_noise)
snr = -120
s += additive_noise / En * Es * 10**(snr/20)

# Desired impulse responses at selected angles
phi_k = np.linspace(0, 2 * np.pi, num=K, endpoint=False)
x_k = [R*np.cos(phi_k), R*np.sin(phi_k), np.zeros_like(phi_k)]
delay_k, weight_k = greens_point(xs, x_k)
waveform_k, shift_k, offset_k = impulse_response(xs, x_k, 'point', fs)
h0, _, _ = construct_ir_matrix(waveform_k, shift_k, N)
H0 = np.fft.rfft(h0, axis=-1)

# System identification
h = system_identification(phi, s, phi_k, p, interpolation='lagrange', int_order=int_order)
H = np.fft.rfft(h, axis=-1)

# Modal Beamforming
freq = np.linspace(0, fs/2, num=H.shape[-1], endpoint=True)
k = 2 * np.pi * freq / c

phi_pwd = np.linspace(0, 2*np.pi, num=360, endpoint=False)

order = 20
Bn = micarray.modal.radial.circular_pw(order, k, R, setup='open')
Dn, _ = micarray.modal.radial.regularize(1/Bn, 100, 'softclip')
D = micarray.modal.radial.circ_diagonal_mode_mat(Dn)
Psi_p = micarray.modal.angular.cht_matrix(order, phi_k, 2*np.pi/len(phi_k))
Psi_q = micarray.modal.angular.cht_matrix(order, phi_pwd)
A_pwd = np.matmul(np.matmul(Psi_q.T, D), np.conj(Psi_p))

q0_pwd = np.squeeze(np.matmul(A_pwd, np.expand_dims(H0.T, 2)))
q0_pwd_t = np.fft.irfft(q0_pwd, axis=0)
q_pwd = np.squeeze(np.matmul(A_pwd, np.expand_dims(H.T, 2)))
q_pwd_t = np.fft.irfft(q_pwd, axis=0)

# Plots
phi_pwd_deg = np.rad2deg(phi_pwd)
freq_kHz = freq / 1000
time_ms = np.arange(N) / fs * 1000

# Fig. Beam pattern - original
plt.figure()
plt.pcolormesh(phi_pwd_deg, freq, db(q0_pwd), vmin=-40, cmap='Blues')
plt.colorbar(label='dB')
plt.xlabel(r'$\phi$ / deg')
plt.ylabel('$f$ / kHz')
plt.title('Modal beamforming - original')

# Fig. Beam pattern - measured
plt.figure()
plt.pcolormesh(phi_pwd_deg, freq, db(q_pwd), vmin=-40, cmap='Blues')
plt.colorbar(label='dB')
plt.xlabel(r'$\phi$ / deg')
plt.ylabel('$t$ / ms')
plt.title('Modal beamforming - measured')

# Fig. PWD signals - original
plt.figure()
plt.pcolormesh(phi_pwd_deg, time_ms, db(q0_pwd_t), vmin=-60, cmap='Blues')
plt.colorbar(label='dB')
plt.xlabel(r'$\phi$ / deg')
plt.ylabel('$f$ / kHz')
plt.title('PWD signals - original')

# Fig. PWD signals - measured
plt.figure()
plt.pcolormesh(phi_pwd_deg, time_ms, db(q_pwd_t), vmin=-60, cmap='Blues')
plt.colorbar(label='dB')
plt.xlabel(r'$\phi$ / deg')
plt.ylabel('$t$ / ms')
plt.title('PWD signals - measured')





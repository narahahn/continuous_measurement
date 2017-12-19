"""
Plane wave decomposition using continuously measured impulse responses

* point source in a free-field
* CARDIOID microphone moving on a circle at a constant speed
* captured signal computed by using fractional delay filters + oversampling
* system identification based on spatial interpolation of a given order
* impulse responses computed for discrete positions
* apply plane wave decomposition to the impulse responses
* compare the results for discrete (sequential) and continuous measurements
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
xs = [0, 6, 0]  # Point source
source_type = 'point'

# Receiver
R = 0.5
Omega = 2 * np.pi / 20
L = int(2 * np.pi / Omega * fs)
t = (1/fs) * np.arange(L)
phi0 = 0
phi = Omega * t + phi0
xm = [R*np.cos(phi), R*np.sin(phi), np.zeros_like(phi)]

# Excitation
N = 1600  # excitation period
p = perfect_sequence_randomphase(N)

# Experimental parameters
K = int(L/N)  # number of target angles = equivalent number of sampling points
int_order = 30  # spatial interpolation order
Omega_al = c / N / R  # anti-aliasing angular speed

# Captured signal of a moving microphone
rm = np.array(xs)[:, np.newaxis] - np.array(xm)
cardioid_gain = 0.5 * (np.sum(rm*xm, axis=0) / np.linalg.norm(rm, axis=0) / np.linalg.norm(xm, axis=0)+1)
waveform_l, shift_l, offset_l = impulse_response(xs, xm, source_type, fs)
waveform_l *= cardioid_gain[:, np.newaxis]
s = captured_signal(waveform_l, shift_l, p)

# Target angles for impulse response computation
phi_k = np.linspace(0, 2 * np.pi, num=K, endpoint=False)

# Static measurement (reference)
x_k = [R*np.cos(phi_k), R*np.sin(phi_k), np.zeros_like(phi_k)]
r_k = np.array(xs)[:, np.newaxis] - np.array(x_k)
cardioid_gain = 0.5 * (np.sum(r_k*x_k, axis=0) / np.linalg.norm(r_k, axis=0) / np.linalg.norm(x_k, axis=0)+1)
waveform_k, shift_k, offset_k = impulse_response(xs, x_k, source_type, fs)
waveform_k *= cardioid_gain[:, np.newaxis]
h0, _, _ = construct_ir_matrix(waveform_k, shift_k, N)
H0 = np.fft.rfft(h0, axis=-1)

# Continuous measurement
h = system_identification(phi, s, phi_k, p, interpolation='lagrange', int_order=int_order)
H = np.fft.rfft(h, axis=-1)


# Modal Beamforming
bf_order = (K-1) // 2  # beamforming order
Npwd = 360
freq = np.linspace(0, fs/2, num=H.shape[-1], endpoint=True)
k = 2 * np.pi * freq / c
phi_pwd = np.linspace(0, 2*np.pi, num=Npwd, endpoint=False)

Bn = micarray.modal.radial.circular_pw(bf_order, k, R, setup='card')
Dn, _ = micarray.modal.radial.regularize(1/Bn, 3000, 'softclip')
D = micarray.modal.radial.circ_diagonal_mode_mat(Dn)
Psi_p = micarray.modal.angular.cht_matrix(bf_order, phi_k, 2*np.pi/len(phi_k))
Psi_q = micarray.modal.angular.cht_matrix(bf_order, phi_pwd)
A_pwd = np.matmul(np.matmul(Psi_q.T, D), np.conj(Psi_p))

q0_pwd = np.squeeze(np.matmul(A_pwd, np.expand_dims(H0.T, 2)))
q0_pwd_t = np.fft.irfft(q0_pwd, axis=0)
q_pwd = np.squeeze(np.matmul(A_pwd, np.expand_dims(H.T, 2)))
q_pwd_t = np.fft.irfft(q_pwd, axis=0)


# Plots
phi_pwd_deg = np.rad2deg(phi_pwd)
freq_kHz = freq / 1000
time_ms = np.arange(N) / fs * 1000
t0 = np.linalg.norm(xs) / c * 1000

# Fig. Beam pattern - discrete
plt.figure(figsize=(10, 8))
plt.pcolormesh(phi_pwd_deg, freq, db(q0_pwd))
plt.colorbar(label='dB')
plt.clim(-100, 0)
plt.xlabel(r'$\phi$ / deg')
plt.ylabel('$f$ / kHz')
plt.title('Modal beamforming - discrete')

# Fig. Beam pattern - continuous
plt.figure(figsize=(10, 8))
plt.pcolormesh(phi_pwd_deg, freq, db(q_pwd))
plt.colorbar(label='dB')
plt.clim(-100, 0)
plt.xlabel(r'$\phi$ / deg')
plt.ylabel('$t$ / ms')
plt.title('Modal beamforming - continuous')

# Fig. PWD signals - discrete
plt.figure(figsize=(10, 8))
plt.pcolormesh(phi_pwd_deg, time_ms, db(q0_pwd_t))
plt.colorbar(label='dB')
plt.clim(-80, 0)
plt.xlabel(r'$\phi$ / deg')
plt.ylabel('$t$ / ms')
#plt.ylim(t0-15, t0+15)
plt.title('PWD signals - discrete')

# Fig. PWD signals - continuous
plt.figure(figsize=(10, 8))
plt.pcolormesh(phi_pwd_deg, time_ms, db(q_pwd_t))
plt.colorbar(label='dB')
plt.clim(-80, 0)
plt.xlabel(r'$\phi$ / deg')
plt.ylabel('$t$ / ms')
#plt.ylim(t0-15, t0+15)
plt.title('PWD signals - continuous')
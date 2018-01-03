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
xs = [0, 3, 3]
source_type = 'point'

# Excitation
N = 800  # excitation period
p = perfect_sequence_randomphase(N)

# Sampling scheme -- Gaussian-like
max_sht_order = 20
x, weights = np.polynomial.legendre.leggauss(max_sht_order+1)
weights *= np.pi / (max_sht_order+1)
theta = np.arccos(x)
channels = max_sht_order + 1
Q = len(theta)
M = 2*max_sht_order + 2

# Microphones
R = 0.2
L = int(N*M)
Omega = 2*np.pi*fs / L
t = (1/fs) * np.arange(L)
phi0 = 0
phi = Omega*t + phi0
positions = np.stack((
        R*np.sin(theta[:, np.newaxis])*np.cos(phi), \
        R*np.sin(theta[:, np.newaxis])*np.sin(phi), \
        R*np.cos(theta[:, np.newaxis])*np.ones_like(phi)), axis=-1)

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
weights *= np.pi / (max_sht_order+1)  # sum(weights)*(2*max_sht_order+2) = 4pi
elev = np.arccos(x)

# Continuous system identification
# (1) Based on CHT for each elevation
h = np.zeros((Q, M, N))
max_cht_order = (int(L/N)-1) // 2
Nf = N // 2 + 1
freq = np.linspace(0, fs/2, num=Nf, endpoint=True)
k = 2 * np.pi * freq / c
Pinv = np.fft.rfft(np.roll(p[::-1], 1))
for q in range(Q): # each elevation
    H_cht = np.zeros((Nf, 2*max_cht_order+1)).astype(complex)
    for i in range(N):
        phi_i = phi[i::N]
        s_i = s[q, i::N]
        P_i = Pinv * np.exp(-1j*2*np.pi/N*np.arange(Nf)*i)
        H_i = P_i[:, np.newaxis] * s_i[np.newaxis, :]
        Psi_i = micarray.modal.angular.cht_matrix(max_cht_order, phi_i, 1/M)
        H_cht += np.matmul(H_i, np.conj(Psi_i.T))
    Psi = micarray.modal.angular.cht_matrix(max_cht_order, azi)
    H_q = np.matmul(H_cht, Psi).T
    h[q, :, :] = np.fft.irfft(H_q, axis=-1)
hrs = np.reshape(h, (Q*M, N))

# (2) SHT
#h2 = np.zeros((Q, M, N))
max_sht_order = (int(L/N)-1) // 2
Nf = N // 2 + 1
freq = np.linspace(0, fs/2, num=Nf, endpoint=True)
k = 2 * np.pi * freq / c
Pinv = np.fft.rfft(np.roll(p[::-1], 1))
H_sht = np.zeros((Nf, (max_sht_order+1)**2)).astype(complex)
for i in range(N):
    phi_i = phi[i::N]
    azi, elev = np.meshgrid(phi_i, theta)
    s_i = s[:, i::N]
    azi = np.matrix.flatten(azi)
    elev = np.matrix.flatten(elev)
    s_i = np.matrix.flatten(s_i)
    P_i = Pinv * np.exp(-1j*2*np.pi/N*np.arange(Nf)*i)
    H_i = P_i[:, np.newaxis] * s_i[np.newaxis, :]
    Y_i = micarray.modal.angular.sht_matrix(max_sht_order, azi, elev, np.repeat(weights, M, axis=0))
    H_sht += np.matmul(H_i, np.conj(Y_i.T))



# Original impulse responses
azi = np.linspace(0, 2*np.pi, 2*max_sht_order+2, endpoint=False)
x, weights = np.polynomial.legendre.leggauss(max_sht_order+1)
elev = np.arccos(x)
h0 = np.zeros_like(h)
for i, theta_i in enumerate(elev):
    x_k = [R*np.sin(theta_i)*np.cos(azi), R*np.sin(theta_i)*np.sin(azi), R*np.cos(theta_i)*np.ones_like(azi)]
    waveform_k, shift_k, offset_k = impulse_response(xs, x_k, source_type, fs)
    htemp, _, _ = construct_ir_matrix(waveform_k, shift_k, N)
#    H0 = np.fft.rfft(h0, axis=-1)
    h0[i, :, :] = htemp
h0rs = np.reshape(h0, (Q*M, N))


# Evaluating individual impulse responses
system_distance = nmse(hrs, h0rs, normalize='mean')

# Monofrequency analysis
f = 2000
H = np.matmul(hrs, np.exp(1j*2*np.pi*f/fs*np.arange(N)))
H0 = np.matmul(h0rs, np.exp(1j*2*np.pi*f/fs*np.arange(N)))
azi, elev, weights = micarray.modal.angular.grid_gauss(max_sht_order)
Yp = micarray.modal.angular.sht_matrix(max_sht_order, azi, elev, weights)
Hnm = np.matmul(np.conj(Yp), H)
H0nm = np.matmul(np.conj(Yp), H0)

# reconstruction (synthesis) on the sphere surfrace
K = 50
azi = np.linspace(0, 2*np.pi, num=K)
elev = np.linspace(np.pi/K, np.pi, num=K)
Azi, Elev = np.meshgrid(azi, elev)
Y = micarray.modal.angular.sht_matrix(max_sht_order, np.ndarray.flatten(Azi), np.ndarray.flatten(Elev))
S = np.squeeze(np.matmul(Y.T, Hnm[:, np.newaxis]))
S = np.reshape(S, (K, K))
S0 = np.squeeze(np.matmul(Y.T, H0nm[:, np.newaxis]))
S0 = np.reshape(S0, (K, K))


# IR reconstruction based on the SHT coefficients
H2 = np.matmul(H_sht, Y)
h2 = np.fft.irfft(H2, n=N, axis=0)


#azi = np.linspace(0, 2*np.pi, 2*max_sht_order+2, endpoint=False)
#x, weights = np.polynomial.legendre.leggauss(max_sht_order+1)
#elev = np.arccos(x)
h0K = np.zeros((K, K, N))
for i, theta_i in enumerate(elev):
    x_k = [R*np.sin(theta_i)*np.cos(azi), R*np.sin(theta_i)*np.sin(azi), R*np.cos(theta_i)*np.ones_like(azi)]
    waveform_k, shift_k, offset_k = impulse_response(xs, x_k, source_type, fs)
    htemp, _, _ = construct_ir_matrix(waveform_k, shift_k, N)
#    H0 = np.fft.rfft(h0, axis=-1)
    h0K[i, :, :] = htemp
h0Krs = np.reshape(h0K, (K*K, N)).T

system_distance = nmse(h2.T, h0Krs.T)
system_distance = np.reshape(system_distance, (K, K))


# Plots
nn = np.random.randint(0, Q*M)
plt.figure()
plt.plot(db(hrs[nn, :]))
plt.plot(db(h0rs[nn, :]))

plt.figure()
plt.plot(db(system_distance))


plt.figure()
plt.pcolormesh(db(h0rs))
plt.colorbar()

plt.figure()
plt.pcolormesh(db(hrs))
plt.colorbar()


plt.figure()
plt.plot(db(H))
plt.plot(db(H0))


plt.figure()
plt.pcolormesh(np.real(S))
plt.colorbar()

plt.figure()
plt.pcolormesh(np.real(S0))
plt.colorbar()

plt.figure()
plt.pcolormesh(np.real(S-S0))
plt.colorbar()



plt.figure()
plt.pcolormesh(db(S))
plt.colorbar(label='dB')

plt.figure()
plt.pcolormesh(db(S0))
plt.colorbar(label='dB')

plt.figure()
plt.pcolormesh(db(S-S0))
plt.colorbar(label='dB')


plt.figure()
plt.imshow(db(h0Krs - h2))
plt.colorbar()

nn = np.random.randint(0, K**2)
plt.figure()
plt.plot(db(h0Krs[:, nn]))
plt.plot(db(h2[:, nn]))
plt.title('{}'.format(nn))

plt.figure()
plt.pcolormesh(np.rad2deg(azi), np.rad2deg(elev), db(system_distance), vmax=0)
plt.colorbar(label='dB')
plt.gca().invert_yaxis()

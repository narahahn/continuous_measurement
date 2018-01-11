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
from sfs.util import sph2cart

# Constants
c = 343
fs = 16000

# Source
alpha = np.pi/2
beta = np.pi/4
xs = sph2cart(np.pi/2, np.pi/4, 3)
source_type = 'point'

# Excitation
N = 800  # excitation period
p = perfect_sequence_randomphase(N)

# Spherical array
R = 0.15

# Gaussian-like sampling scheme
modal_bandwidth = int(np.ceil(np.exp(1)*np.pi*fs/2*R/c))
max_sht_order = 2
x, weights = np.polynomial.legendre.leggauss(max_sht_order+1)
weights *= np.pi / (max_sht_order+1)
theta = np.arccos(x)
channels = max_sht_order + 1
N_theta = len(theta)
N_phi = 2*max_sht_order + 2

# Circular trajectory
L = int(N*N_phi)
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
s = np.zeros((N_theta, L))
for q in range(len(theta)):
    xm = np.squeeze(positions[q, :, :])
    waveform_l, shift_l, offset_l = impulse_response(xs, xm, source_type, fs)
    s[q, :] = captured_signal(waveform_l, shift_l, p)
    s[q, :] += additive_noise(s[q, :], snr)

# Continuous system identification
# (1) Based on CHT for each elevation
#h = np.zeros((Q, M, N))
#max_cht_order = (int(L/N)-1) // 2
#Nf = N // 2 + 1
#freq = np.linspace(0, fs/2, num=Nf, endpoint=True)
#k = 2 * np.pi * freq / c
#Pinv = np.fft.rfft(np.roll(p[::-1], 1))
#for q in range(Q): # each elevation
#    H_cht = np.zeros((Nf, 2*max_cht_order+1)).astype(complex)
#    for i in range(N):
#        phi_i = phi[i::N]
#        s_i = s[q, i::N]
#        P_i = Pinv * np.exp(-1j*2*np.pi/N*np.arange(Nf)*i)
#        H_i = P_i[:, np.newaxis] * s_i[np.newaxis, :]
#        Psi_i = micarray.modal.angular.cht_matrix(max_cht_order, phi_i, 1/M)
#        H_cht += np.matmul(H_i, np.conj(Psi_i.T))
#    Psi = micarray.modal.angular.cht_matrix(max_cht_order, azi)
#    H_q = np.matmul(H_cht, Psi).T
#    h[q, :, :] = np.fft.irfft(H_q, axis=-1)
#hrs = np.reshape(h, (Q*M, N))

# Spherical harmonics transform
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
    Y_i = micarray.modal.angular.sht_matrix(max_sht_order, azi, elev, np.repeat(weights, N_phi, axis=0))
    H_sht += np.matmul(H_i, np.conj(Y_i.T))

# Target positions
oversample_order = 50
_, _, weights = micarray.modal.angular.grid_gauss(oversample_order)
phi_target = np.linspace(0, 2*np.pi, 2*oversample_order+2, endpoint=False)
x, weights = np.polynomial.legendre.leggauss(oversample_order+1)
weights *= np.pi / (oversample_order+1)  # sum(weights)*(2*max_sht_order+2) = 4pi
theta_target = np.arccos(x)
Q_theta = len(theta_target)
Q_phi = len(phi_target)

# Original impulse responses
h0_2d = np.zeros((Q_theta, Q_phi, N))
for i, theta_i in enumerate(theta_target):
    x_k = [R*np.sin(theta_i)*np.cos(phi_target), R*np.sin(theta_i)*np.sin(phi_target), R*np.cos(theta_i)*np.ones_like(phi_target)]
    waveform_k, shift_k, offset_k = impulse_response(xs, x_k, source_type, fs)
    htemp, _, _ = construct_ir_matrix(waveform_k, shift_k, N)
#    H0 = np.fft.rfft(h0, axis=-1)
    h0_2d[i, :, :] = htemp
h0 = np.reshape(h0_2d, (Q_theta*Q_phi, N))


# Inverse spherical harmonics transform
Phi_target, Theta_target = np.meshgrid(phi_target, theta_target)
Y = micarray.modal.angular.sht_matrix(max_sht_order, np.ndarray.flatten(Phi_target), np.ndarray.flatten(Theta_target))
H = np.matmul(H_sht, Y)
h = np.fft.irfft(H, n=N, axis=0).T
h_2d = np.reshape(h, (Q_theta, Q_phi, N))

# Evaluation
E = nmse(h, h0, 'each')
E_matrix = np.reshape(E, (Q_theta, Q_phi))
E_mean = np.mean(E)

Y00 = micarray.modal.angular.sht_matrix(0, np.ndarray.flatten(Phi_target), np.ndarray.flatten(Theta_target), np.repeat(weights, Q_phi, axis=0))
Yinv = micarray.modal.angular.sht_matrix(0, np.ndarray.flatten(Phi_target), np.ndarray.flatten(Theta_target))
E00 = np.real(np.matmul(E, np.conj(Y00.T))[0]) / np.sqrt(4*np.pi)

# Evaluating individual impulse responses
#system_distance = nmse(hrs, h0_reshape, normalize='mean')

# Monofrequency analysis
#f = 2000
#H = np.matmul(hrs, np.exp(1j*2*np.pi*f/fs*np.arange(N)))
#H0 = np.matmul(h0rs, np.exp(1j*2*np.pi*f/fs*np.arange(N)))
#azi, elev, weights = micarray.modal.angular.grid_gauss(max_sht_order)
#Yp = micarray.modal.angular.sht_matrix(max_sht_order, azi, elev, weights)
#Hnm = np.matmul(np.conj(Yp), H)
#H0nm = np.matmul(np.conj(Yp), H0)

# reconstruction (synthesis) on the sphere surfrace
#K = 100
#azi = np.linspace(0, 2*np.pi, num=K)
#elev = np.linspace(np.pi/K, np.pi, num=K)
#Azi, Elev = np.meshgrid(azi, elev)
#Y = micarray.modal.angular.sht_matrix(max_sht_order, np.ndarray.flatten(Azi), np.ndarray.flatten(Elev))
#S = np.squeeze(np.matmul(Y.T, Hnm[:, np.newaxis]))
#S = np.reshape(S, (K, K))
#S0 = np.squeeze(np.matmul(Y.T, H0nm[:, np.newaxis]))
#S0 = np.reshape(S0, (K, K))

#azi = np.linspace(0, 2*np.pi, 2*max_sht_order+2, endpoint=False)
#x, weights = np.polynomial.legendre.leggauss(max_sht_order+1)
#elev = np.arccos(x)
#h0K = np.zeros((K, K, N))
#for i, theta_i in enumerate(elev):
#    x_k = [R*np.sin(theta_i)*np.cos(azi), R*np.sin(theta_i)*np.sin(azi), R*np.cos(theta_i)*np.ones_like(azi)]
#    waveform_k, shift_k, offset_k = impulse_response(xs, x_k, source_type, fs)
#    htemp, _, _ = construct_ir_matrix(waveform_k, shift_k, N)
#    H0 = np.fft.rfft(h0, axis=-1)
#    h0K[i, :, :] = htemp
#h0Krs = np.reshape(h0K, (K*K, N)).T

dirname = 'data_aes144'
filename = 'msr_N{:04d}_order{:03d}'.format(N, max_sht_order)
np.savez('{}/{}'.format(dirname,filename),
         fs=fs, c=c,
         alpha=alpha, beta=beta,
         modal_bandwidth=modal_bandwidth, max_sht_order=max_sht_order,
         oversample_order=oversample_order,
         R=R, N_theta=N_theta, N_phi=N_phi, theta=theta, phi=phi, L=L, N=N,
         Omega=Omega, snr=snr,
         phi_target=phi_target, theta_target=theta_target,
         Q_theta=Q_theta, Q_phi=Q_phi,
         h0=h0, h=h, E_matrix=E_matrix, E_mean=E_mean, E00=E00
         )


# Plots
nn = np.random.randint(0, h.shape[1])
plt.figure()
plt.plot(db(h[nn, :]))
plt.plot(db(h0[nn, :]))
plt.ylim(ymin=-120)
plt.title('{}'.format(nn))

plt.figure()
plt.plot(h[nn, :])
plt.plot(h0[nn, :])
plt.title('{}'.format(nn))

qq = np.random.randint(0, Q_theta)
plt.figure()
plt.imshow(db(h0_2d[qq, :, :] - h_2d[qq, :, :]).T, vmax=0)
plt.colorbar(label='dB')
plt.axis('normal')



plt.figure(figsize=(8, 4))
plt.pcolormesh(np.rad2deg(phi_target), np.rad2deg(theta_target), db(E_matrix))
plt.colorbar(label='dB')
plt.plot(np.rad2deg(alpha), np.rad2deg(beta), 'x')
plt.gca().invert_yaxis()
plt.axis('equal')
plt.xticks(np.arange(0, 360+90, 90))
plt.yticks(np.arange(0, 180+90, 90))
plt.xlim(0, 360)
plt.ylim(0, 180)
plt.title('Mean NMSE = {} .. {}'.format(db(E_mean), db(E00)))


plt.figure(figsize=(8, 4))
plt.imshow(db(E_matrix), vmax=0, 
           extent=[np.rad2deg(phi_target[0]), np.rad2deg(phi_target[-1]), np.rad2deg(theta_target[-1]), np.rad2deg(theta_target[0])])
plt.plot(np.rad2deg(alpha), np.rad2deg(beta), 'x')
#plt.xticks(np.arange(0, 360+90, 60))
#plt.yticks(np.arange(0, 180+90, 60))
plt.ylim(0, 180)
plt.xlim(0, 360)
plt.axis('tight')
plt.colorbar(label='dB')
plt.gca().invert_yaxis()
plt.title('Mean NMSE = {}'.format(db(E_mean)))


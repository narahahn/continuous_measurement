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
from utils import *
from source import *
from sfs.util import sph2cart
from sys import path
path.append('../')

# Constants
c = 343
fs = 16000

# Source
rho = 3
alpha = np.pi/2
beta = np.pi/4
xs = sph2cart(alpha, beta, rho)
source_type = 'point'

# Excitation
N = 800  # excitation period
p = perfect_sequence_randomphase(N)

# Spherical array
R = 0.15

# Gaussian-like sampling scheme
modal_bandwidth = int(np.ceil(np.exp(1)*np.pi*fs/2*R/c))
max_sht_order = 15
x, weights = np.polynomial.legendre.leggauss(max_sht_order+1)
weights *= np.pi / (max_sht_order+1)
theta = np.arccos(x)
channels = max_sht_order + 1
N_theta = len(theta)
N_phi = 2*max_sht_order + 2

# Azimuth angle
L = int(N*N_phi)
Omega = 2*np.pi*fs / L
t = (1/fs) * np.arange(L)
phi0 = 0
phi = Omega*t + phi0
positions = np.stack((
        R*np.sin(theta[:, np.newaxis])*np.cos(phi),
        R*np.sin(theta[:, np.newaxis])*np.sin(phi),
        R*np.cos(theta[:, np.newaxis])*np.ones_like(phi)), axis=-1)

# Captured signal and additive noise
snr = -np.infty
s = np.zeros((N_theta, L))
for q in range(len(theta)):
    xm = np.squeeze(positions[q, :, :])
    waveform_l, shift_l, offset_l = impulse_response(xs, xm, source_type, fs)
    s[q, :] = captured_signal(waveform_l, shift_l, p)
    s[q, :] += additive_noise(s[q, :], snr)

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
    Y_i = micarray.modal.angular.sht_matrix(
            max_sht_order,
            azi, elev,
            np.repeat(weights, N_phi, axis=0))
    H_sht += np.matmul(H_i, np.conj(Y_i.T))

# Target positions
oversample_order = 50
_, _, weights = micarray.modal.angular.grid_gauss(oversample_order)
phi_target = np.linspace(0, 2*np.pi, 2*oversample_order+2, endpoint=False)
x, weights = np.polynomial.legendre.leggauss(oversample_order+1)
weights *= np.pi / (oversample_order+1)  # sum(weights)*(2*max_sht_order+2)=4pi
theta_target = np.arccos(x)
Q_theta = len(theta_target)
Q_phi = len(phi_target)

# Original impulse responses
h0_2d = np.zeros((Q_theta, Q_phi, N))
for i, theta_i in enumerate(theta_target):
    x_k = [R*np.sin(theta_i)*np.cos(phi_target),
           R*np.sin(theta_i)*np.sin(phi_target),
           R*np.cos(theta_i)*np.ones_like(phi_target)]
    waveform_k, shift_k, offset_k = impulse_response(xs, x_k, source_type, fs)
    htemp, _, _ = construct_ir_matrix(waveform_k, shift_k, N)
    h0_2d[i, :, :] = htemp
h0 = np.reshape(h0_2d, (Q_theta*Q_phi, N))

# Continuous measurement - Inverse spherical harmonics transform
Phi_target, Theta_target = np.meshgrid(phi_target, theta_target)
Y = micarray.modal.angular.sht_matrix(
        max_sht_order,
        np.ndarray.flatten(Phi_target),
        np.ndarray.flatten(Theta_target))
H = np.matmul(H_sht, Y)
h = np.fft.irfft(H, n=N, axis=0).T
h_2d = np.reshape(h, (Q_theta, Q_phi, N))

# Evaluation
E = nmse(h, h0, 'each')
E_matrix = np.reshape(E, (Q_theta, Q_phi))
E_mean = np.mean(E)

# Mean NMSE
Y00 = micarray.modal.angular.sht_matrix(
        0,
        np.ndarray.flatten(Phi_target),
        np.ndarray.flatten(Theta_target),
        np.repeat(weights, Q_phi, axis=0))
E00 = np.real(np.matmul(E, np.conj(Y00.T))[0]) / np.sqrt(4*np.pi)

# Save simulation results
dirname = 'data_aes144'
filename = 'msr_N{:04d}_order{:03d}'.format(N, max_sht_order)
np.savez('{}/{}'.format(dirname, filename),
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

# Impulse responses - dB
nn = np.random.randint(0, h.shape[0])
q_theta_pick, q_phi_pick = np.divmod(nn, Q_phi)
theta_pick = np.rad2deg(theta_target[q_theta_pick])
phi_pick = np.rad2deg(phi_target[q_phi_pick])

time = np.arange(N)/fs*1000
plt.figure()
plt.plot(time, db(h0[nn, :]), label='static')
plt.plot(time, db(h[nn, :]), label='dynamic')
plt.ylim(ymin=-120)
plt.xlabel('$t$ / ms')
plt.ylabel('Amplitude / dB')
plt.legend(loc='best')
plt.title(r'$\theta={:.1f}^\circ$, $\phi={:.1f}^\circ$'
          .format(theta_pick, phi_pick))

# Impulse responses - linear
plt.figure()
plt.plot(time, h0[nn, :], label='static')
plt.plot(time, h[nn, :], label='dynamic')
plt.xlabel('$t$ / ms')
plt.ylabel('Amplitude / dB')
plt.xlim(5, 15)
plt.legend(loc='best')
plt.title(r'$\theta={:.1f}^\circ$, $\phi={:.1f}^\circ$'
          .format(theta_pick, phi_pick))

# Filter coefficient errors - dB
qq = np.random.randint(0, Q_theta)
theta_pick = np.rad2deg(theta_target[qq])
plt.figure()
plt.pcolormesh(np.rad2deg(phi_target), time,
               db(h0_2d[qq, :, :]-h_2d[qq, :, :]).T, vmax=0)
plt.colorbar(label='dB')
plt.axis('normal')
plt.xlabel('$\phi$ / deg')
plt.ylabel('time / ms')
plt.title(r'$\theta={:.1f}^\circ$'.format(theta_pick))

# NMSE map - dB
plt.figure()
plt.pcolormesh(np.rad2deg(phi_target), np.rad2deg(theta_target), db(E_matrix))
plt.colorbar(label='dB')
plt.plot(np.rad2deg(alpha), np.rad2deg(beta), 'x')
plt.gca().invert_yaxis()
plt.axis('tight')
plt.xlabel('$\phi$ / deg')
plt.ylabel('$\theta$ / deg')
plt.title('Mean NMSE = {0:.1f} dB'.format(db(E00)))

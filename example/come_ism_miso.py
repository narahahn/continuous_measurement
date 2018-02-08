"""
Continuous measurement of room impulse responses using a moving microphone.

* multiple point sources in a rectanular room
* impulse responses simulated with the image source method 
* omnidirectional microphone moving on a circle at a constant speed
* captured signal computed by using fractional delay filters + oversampling
* system identification based on spatial interpolation of a given order
"""
import numpy as np
import matplotlib.pyplot as plt
from sys import path
from utils import (perfect_sweep, perfect_sequence_randomphase,
                   captured_signal, system_identification,
                   additive_noise, nmse, next_divisor)
from source import impulse_response
from sfs.util import image_sources_for_box, sph2cart, db
from matplotlib.patches import Rectangle
path.append('../')

# Constants
c = 343
fs = 16000

# Room
dimension = 5.8, 5.0, 3.0  # room dimensions
coeff = .5, .5, .5, .5, .5, .5  # wall reflection coefficients
room_center = np.array([2.9, 2.5, 1.5])

# Point sources (loudspeaker array)
loudspeaker_array = np.genfromtxt('./data_eusipco2018/university_rostock.csv', delimiter=',')
idx = 56, 61, 2, 7
M = len(idx)
loudspeaker_array = loudspeaker_array[idx, :3] + room_center
source_type = 'point'

# 3D Image sources
max_order = 3  # maximum order of image sources
image_sources = []
source_strength = []
for m in range(M):
    x0 = loudspeaker_array[m]
    xs, wall_count = image_sources_for_box(x0, dimension, max_order)
    image_sources.append(xs)
    source_strength.append(np.prod(coeff**wall_count, axis=1))

# Receiver
array_center = room_center
R = 0.2  # radius

# Moving microphone
Omega = 2 * np.pi / 15
L = int(np.ceil(2 * np.pi / Omega * fs))
t = 1 / fs * np.arange(L)
phi0 = 0
phi = Omega * t + phi0
theta = np.pi / 2
x = sph2cart(phi, theta, R) + array_center
x = np.array([x[0], x[1], x[2] * np.ones(L)]).T

# Perfect sequence
max_delay = (np.max(dimension) * (max_order + 1)) / c * fs
Nir = next_divisor(max_delay, fs)
N = M * Nir
p = perfect_sequence_randomphase(N)
#p = perfect_sweep(N)

# Anti-aliasing condition
Omega_al = c / N / R * 0.8 # anti-aliasing angular speed

# Target positions
K_target = 360  # number of microphones
phi_target = np.linspace(0, 2 * np.pi, num=K_target, endpoint=False)
theta_target = np.pi / 2
x_target = sph2cart(phi_target, theta_target, R) + array_center
x_target = np.array([x_target[0], x_target[1], x_target[2] * np.ones(K_target)]).T

# Desired impulse responses
h0 = []
for m in range(M):
    htemp = np.zeros((K_target, Nir))
    xs = image_sources[m]
    scale = source_strength[m]
    for i, source_pos in enumerate(xs):
        waveform, shift, _ = impulse_response(source_pos, x_target, source_type, fs)
        Lw = waveform.shape[1]
        waveform *= scale[i]
        for k in range(K_target):
            htemp[k, shift[k]:shift[k]+Lw] += waveform[k]
    h0.append(htemp)
h0 = np.array(h0)

# Captured signal
snr = -np.infty
s = 0
for m in range(M):
    xs = image_sources[m]
    scale = source_strength[m]
    for i, source_pos in enumerate(xs):
        waveform, shift, _ = impulse_response(source_pos, x, source_type, fs)
        s += captured_signal(waveform * scale[i], shift, np.roll(p, int(m * Nir)))
s += additive_noise(s, snr)


# System identification
int_order = 15
int_type = 'lagrange'
int_type = 'cht'
hhat = system_identification(phi, s, phi_target, p,
                             interpolation=int_type, int_order=int_order)
hhat = np.concatenate([hhat[np.newaxis, :, i * Nir: (i+1) * Nir] for i in range(M)], axis=0)


# Plots

# Axes
mm = np.random.randint(M)
nn = np.random.randint(K_target)
tau = np.arange(Nir) / fs * 1000
Nf = int(np.ceil(Nir / 2 + 1))
freq = np.arange(Nf) / Nf * fs / 2
order = np.arange((-K_target)//2, K_target//2)

# Fig. Configuration
plt.figure(figsize=(8, 8))
plt.plot(x_target[:, 0], x_target[:, 1], 'g.')
plt.plot(loudspeaker_array[:, 0], loudspeaker_array[:, 1], 'co')
for m in range(M):
    plt.text(loudspeaker_array[m, 0], loudspeaker_array[m, 1], '{}'.format(idx[m]), va='center')
plt.gca().add_patch(Rectangle((0, 0), dimension[0], dimension[1], fill=False))
plt.xlabel('x / m')
plt.ylabel('y / m')
plt.axis('equal')
plt.title('Configuration (xy-plane)')

# Fig. Impulse response (for a randomly selected angle)
plt.figure()
plt.plot(tau, h0[mm, nn, :], label='original', linewidth=5, color='lightgray')
plt.plot(tau, hhat[mm, nn, :], label='estimate')
plt.legend(loc='best')
plt.xlabel(r'$t$ / ms')
plt.title('Impulse Response (Loudspeaker #{}, $\phi={}^\circ$)'.format(idx[mm], 360 * nn / K_target))

# Fig. Impulse response in dB (for a randomly selected angle)
plt.figure()
plt.plot(tau, db(h0[mm, nn, :]), label='original', linewidth=5, color='lightgray')
plt.plot(tau, db(hhat[mm, nn, :]), label='estimate')
plt.xlabel(r'$t$ / ms')
plt.ylabel('Amplitude / dB')
plt.legend(loc='best')
plt.title('Impulse Response (Loudspeaker #{}, $\phi={}^\circ$)'.format(idx[mm], 360 * nn / K_target))

# Fig. Transfer function (for a randomly selected angle)
plt.figure()
plt.plot(freq, db(np.fft.rfft(h0[mm, nn, :])),
         label='original',
         linewidth=5,
         color='lightgray')
plt.plot(freq, db(np.fft.rfft(hhat[mm, nn, :])), label='estimate')
plt.legend(loc='best')
plt.xscale('log')
plt.xlim(20, fs/2)
plt.ylim(-60, 0)
plt.xlabel(r'$f$ / Hz')
plt.ylabel('Magnitude / dB')
plt.title('Transfer Function (Loudspeaker #{}, $\phi={}^\circ$)'.format(idx[mm], 360 * nn / K_target))

# Fig. Impulse response coefficients in dB
plt.figure()
plt.pcolormesh(np.rad2deg(phi_target), tau, db(hhat[mm]).T, vmin=-100)
plt.axis('normal')
cb = plt.colorbar(label='dB')
#plt.clim(-100, 0)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$\tau$ / ms')
plt.xlim(0, 360)
plt.ylim(0, Nir / fs * 1000)
plt.title('FIR Coefficients (Loudspeaker #{})'.format(idx[mm]))

# Fig. Impulse response coefficients in linear scale
plt.figure()
plt.pcolormesh(np.rad2deg(phi_target), tau, hhat[mm].T, cmap='coolwarm')
plt.axis('normal')
cb = plt.colorbar(label='dB')
plt.clim(-0.02, 0.02)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$\tau$ / ms')
plt.xlim(0, 360)
plt.ylim(0, Nir / fs * 1000)
plt.ylim(0, 20)
plt.title('FIR Coefficients (Loudspeaker #{})'.format(idx[mm]))


# Fig. Impulse response coefficient errors in dB
plt.figure()
plt.pcolormesh(np.rad2deg(phi_target), tau, db(h0[mm] - hhat[mm]).T)
plt.axis('normal')
cb = plt.colorbar(label='dB')
plt.clim(-200, 0)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$\tau$ / ms')
plt.xlim(0, 360)
plt.ylim(0, Nir / fs * 1000)
plt.title('FIR Coefficient Error (Loudspeaker #{}'.format(idx[mm]))

# Fig. Spectral distortion in dB
plt.figure()
plt.pcolormesh(np.rad2deg(phi_target), freq / 1000,
               db(np.fft.rfft(h0[mm] - hhat[mm], axis=-1)).T)
plt.axis('normal')
plt.colorbar(label='dB')
plt.clim(-200, 0)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$f$ / kHz')
plt.xlim(0, 360)
plt.ylim(0, fs / 2 / 1000)
plt.title('Spectral Distortion (Loudspeaker #{})'.format(idx[mm]))

# Fig. Normalized system distance in dB
plt.figure()
for m in range(M):
    plt.plot(np.rad2deg(phi_target), db(nmse(hhat[m], h0[m])))
plt.xlim(0, 360)
plt.ylim(-200, 0)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel('NMSE / dB')
plt.title('Normalized Mean Square Error (Loudspeaker #{})'.format(idx[mm]))

# Fig. Desired CHT spectrum
plt.figure(figsize=(10, 4))
plt.pcolormesh(order, freq / 1000,
               db(np.fft.fftshift(np.fft.fft2(h0[mm]), axes=0)[:, :Nf]).T,
               vmin=-120)
plt.axis('normal')
plt.xlabel('CHT order')
plt.ylabel(r'$f$ / kHz')
plt.colorbar(label='dB')
plt.title('CHT spectrum - original (Loudspeaker #{})'.format(idx[mm]))

# Fig. CHT spectrum of the impulse responses
plt.figure(figsize=(10, 4))
plt.pcolormesh(order, freq / 1000,
               db(np.fft.fftshift(np.fft.fft2(hhat[mm]), axes=0)[:, :Nf]).T,
               vmin=-120)
plt.axis('normal')
plt.xlabel('CHT order')
plt.ylabel(r'$f$ / kHz')
plt.colorbar(label='dB')
plt.title('CHT spectrum - measured (Loudspeaker #{})'.format(idx[mm]))

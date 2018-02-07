"""
Continuous measurement of room impulse responses using a moving microphone.

* point source in a rectanular room
* impulse responses imulated with the image source method 
* omnidirectional microphone moving on a circle at a constant speed
* captured signal computed by using fractional delay filters + oversampling
* system identification based on spatial interpolation of a given order
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from sys import path
path.append('../')
from utils import perfect_sweep, perfect_sequence_randomphase, captured_signal, system_identification, additive_noise
from source import impulse_response
from sfs.util import image_sources_for_box, sph2cart, as_xyz_components, db
from matplotlib.patches import Rectangle

# Constants
c = 343
fs = 16000

# Room
dimension = 5.8, 5.0, 3.0  # room dimensions
coeff = .5, .5, .5, .5, .5, .5  # wall reflection coefficients

# Point source
x0 = 2.9, 4.0, 1.2  # source position
source_type = 'point'

# 3D Image sources
max_order = 1  # maximum order of image sources
xs, wall_count = image_sources_for_box(x0, dimension, max_order)
source_strength = np.prod(coeff**wall_count, axis=1)

# Receiver
array_center = np.array([2.9, 2.5, 1.5])
R = 0.2  # radius

# Moving microphone
Omega = 2 * np.pi / 8
L = int(np.ceil(2 * np.pi / Omega * fs))
t = 1 / fs * np.arange(L)
phi0 = 0
phi = Omega * t + phi0
theta = np.pi / 2
x = sph2cart(phi, theta, R) + array_center
x = np.array([x[0], x[1], x[2] * np.ones(L)]).T

# Perfect sequence
N = int(2**(np.ceil(np.log2((np.max(np.linalg.norm(xs - array_center, axis=-1)) + R) / c * fs))))
N = 1600
p = perfect_sequence_randomphase(N)
#p = perfect_sweep(N)

# Anti-aliasing condition
Omega_al = c / N / R  # anti-aliasing angular speed

# Target positions
K_target = 360  # number of microphones
phi_target = np.linspace(0, 2 * np.pi, num=K_target, endpoint=False)
theta_target = np.pi / 2
x_target = sph2cart(phi_target, theta_target, R) + array_center
x_target = np.array([x_target[0], x_target[1], x_target[2] * np.ones(K_target)]).T

# Desired impulse responses
h0 = np.zeros((K_target, N))
for i, source_pos in enumerate(xs):
    waveform, shift, _ = impulse_response(source_pos, x_target, source_type, fs)
    Lw = waveform.shape[1]
    waveform *= source_strength[i]
    for k in range(K_target):
        h0[k, shift[k]:shift[k]+Lw] += waveform[k]

# Spatial interpolatoin
int_order = 15

# Captured signal
s = 0
for i, source_pos in enumerate(xs):
    waveform, shift, _ = impulse_response(source_pos, x, source_type, fs)
    s += captured_signal(waveform * source_strength[i], shift, p)
snr = -np.infty
s += additive_noise(s, snr)


# System identification
hhat = system_identification(phi, s, phi_target, p,
                             interpolation='cht')


# Plots

# Axes
nn = np.random.randint(0, K_target)
tau = np.arange(N) / fs * 1000
Nf = int(np.ceil(N / 2 + 1))
freq = np.arange(Nf) / Nf * fs / 2
order = np.arange((-K_target)//2, K_target//2)

# Fig. Image sources
plt.figure()
plt.scatter(*xs.T, source_strength * 20)
plt.plot(x[:, 0], x[:, 1], 'g.')
plt.gca().add_patch(Rectangle((0, 0), dimension[0], dimension[1], fill=False))
plt.xlabel('x / m')
plt.ylabel('y / m')
plt.axis('equal')
plt.title('Image sources (xy-plane)')

# Fig. Impulse response (for a randomly selected angle)
plt.figure()
plt.plot(tau, h0[nn, :], label='original', linewidth=5, color='lightgray')
plt.plot(tau, hhat[nn, :], label='estimate')
plt.legend(loc='best')
plt.xlabel(r'$t$ / ms')
plt.title('Impulse Response ($\phi={}^\circ$)'.format(360 * nn / K_target))

# Fig. Impulse response in dB (for a randomly selected angle)
plt.figure()
plt.plot(tau, db(h0[nn, :]), label='original', linewidth=5, color='lightgray')
plt.plot(tau, db(hhat[nn, :]), label='estimate')
plt.xlabel(r'$t$ / ms')
plt.ylabel('Amplitude / dB')
plt.legend(loc='best')
plt.title('Impulse Response ($\phi={}^\circ$)'.format(360 * nn / K_target))

# Fig. Transfer function (for a randomly selected angle)
plt.figure()
plt.plot(freq, db(np.fft.rfft(h0[nn, :])),
         label='original',
         linewidth=5,
         color='lightgray')
plt.plot(freq, db(np.fft.rfft(hhat[nn, :])), label='estimate')
plt.legend(loc='best')
plt.xscale('log')
plt.xlim(20, fs/2)
plt.ylim(-60, 0)
plt.xlabel(r'$f$ / Hz')
plt.ylabel('Magnitude / dB')
plt.title('Transfer Function ($\phi={}^\circ$)'.format(360 * nn / K_target))

# Fig. Impulse response coefficients in dB
plt.figure()
plt.pcolormesh(np.rad2deg(phi_target), tau, db(hhat).T, vmin=-100)
plt.axis('normal')
cb = plt.colorbar(label='dB')
#plt.clim(-100, 0)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$\tau$ / ms')
plt.xlim(0, 360)
plt.ylim(0, N / fs * 1000)
plt.title('FIR Coefficients')

# Fig. Impulse response coefficients in linear scale
plt.figure()
plt.pcolormesh(np.rad2deg(phi_target), tau, hhat.T, cmap='coolwarm')
plt.axis('normal')
cb = plt.colorbar(label='dB')
plt.clim(-0.02, 0.02)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$\tau$ / ms')
plt.xlim(0, 360)
plt.ylim(0, N / fs * 1000)
plt.ylim(0, 20)
plt.title('FIR Coefficients')


# Fig. Impulse response coefficient errors in dB
plt.figure()
plt.pcolormesh(np.rad2deg(phi_target), tau, db(h0 - hhat).T)
plt.axis('normal')
cb = plt.colorbar(label='dB')
plt.clim(-100, 0)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$\tau$ / ms')
plt.xlim(0, 360)
plt.ylim(0, N / fs * 1000)
plt.title('FIR Coefficient Error')

# Fig. Spectral distortion in dB
plt.figure()
plt.pcolormesh(np.rad2deg(phi_target), freq / 1000,
               db(np.fft.rfft(h0 - hhat, axis=-1)).T)
plt.axis('normal')
plt.colorbar(label='dB')
plt.clim(-100, 0)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel(r'$f$ / kHz')
plt.xlim(0, 360)
plt.ylim(0, fs / 2 / 1000)
plt.title('Spectral Distortion')

# Fig. Normalized system distance in dB
plt.figure()
plt.plot(np.rad2deg(phi_target),
         db(np.sqrt(np.sum((h0 - hhat)**2, axis=-1) / np.sum(h0**2, axis=-1))))
plt.xlim(0, 360)
plt.ylim(-120, 0)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel('NMSE / dB')
plt.title('Normalized Mean Square Error')

# Fig. Desired CHT spectrum
plt.figure(figsize=(10, 4))
plt.pcolormesh(order, freq / 1000,
               db(np.fft.fftshift(np.fft.fft2(h0), axes=0)[:, :Nf]).T,
               vmin=-120)
plt.axis('normal')
plt.xlabel('CHT order')
plt.ylabel(r'$f$ / kHz')
plt.colorbar(label='dB')
plt.title('CHT spectrum - original')

# Fig. CHT spectrum of the impulse responses
plt.figure(figsize=(10, 4))
plt.pcolormesh(order, freq / 1000,
               db(np.fft.fftshift(np.fft.fft2(hhat), axes=0)[:, :Nf]).T,
               vmin=-120)
plt.axis('normal')
plt.xlabel('CHT order')
plt.ylabel(r'$f$ / kHz')
plt.colorbar(label='dB')
plt.title('CHT spectrum - measured')

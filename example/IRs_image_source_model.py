""" Computes the impulse responses in a rectangular room using the 
    mirror image sources model
    * frequency-independent reflection coefficients
    * fractional delay interpolation using the Lagrange polynomial
"""
import numpy as np
import sfs
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.signal import lfilter, freqz, bessel, butter, fftconvolve
from matplotlib.patches import Rectangle
from sys import path
path.append('../')
from utils import *
from source import *

# Constants
c = 343
fs = 44100  # sample rate for boadband response

# Room
L = 10, 10.7, 12  # room dimensions
coeffs = .8, .8, .7, .7, .6, .6  # wall reflection coefficients

# Point source
x0 = 5.2, 5.7, 2.5  # source position
signal = ([1, 0, 0], fs)  # signal for broadband response
source_type = 'point'

# 3D Image sources
max_order = 6  # maximum order of image sources
xs, wall_count = sfs.util.image_sources_for_box(x0, L, max_order)
source_strength = np.prod(coeffs**wall_count, axis=1)

# Circular microphone array
R_mic = 0.2  # radius
K_mic = 90  # number of microphones
phi_mic = np.linspace(0, 2 * np.pi, num=K_mic, endpoint=False)
x_mic = np.array([R_mic * np.cos(phi_mic) + 0.732,
                  R_mic * np.sin(phi_mic) + 0.831,
                  np.zeros_like(phi_mic) + 0.511]).T

# Impulse responses
#N = 4800  # FIR filter length
N = int(2**(np.ceil(np.log2(np.max(np.linalg.norm(xs, axis=-1)) / c * fs))))
h = np.zeros((K_mic, N))
for ii, xi in enumerate(xs):
    waveform, shift, offset = impulse_response(xi, x_mic, source_type, fs)
    htemp, _, _ = construct_ir_matrix(waveform, shift, N)
    h += htemp * source_strength[ii]

# Listening example
s, _ = sf.read('50.flac')  # SQAM male speech
s = s[:3*fs, 0]
y = fftconvolve(s, h[0, :])


# Plots
phi_deg = np.rad2deg(phi_mic)
time = np.arange(N)/fs*1000
Nf = N//2+1
freq = np.arange(Nf)*fs/(Nf)

# IRs - linear scale
plt.figure(figsize=(4, 10))
plt.pcolormesh(phi_deg, time, h.T, cmap='coolwarm')
plt.colorbar()
plt.clim(-0.002, 0.002)
plt.axis('tight')
plt.xlabel(r'$\phi$ / deg')
plt.ylabel('$t$ / ms')
plt.ylim(0, 100)

# IRs - dB
plt.figure(figsize=(4, 10))
plt.pcolormesh(phi_deg, time, db(h.T), cmap='Blues')
plt.colorbar()
plt.clim(-200, 0)
plt.xlabel(r'$\phi$ / deg')
plt.ylabel('$t$ / ms')
plt.axis('tight')

# Randomly selected IR - linear scale
nn = np.random.randint(K_mic)
plt.figure()
plt.plot(time, h[nn, :])
plt.xlabel('$t$ / ms')
#plt.ylim(-1, 1)

# Randomly selected IR - dB
plt.figure()
plt.plot(time, db(h[nn, :]))
plt.xlabel('$t$ / ms')
plt.ylim(-120, 0)

# Frequency response
plt.figure()
plt.semilogx(freq, db(np.fft.rfft(h[nn, :])))
plt.ylim(-60, 0)
plt.xlabel('$f$ / Hz')
plt.ylabel('Magnitude / dB')

# Spectrogram
plt.figure()
plt.specgram(h[nn, :], NFFT=128, noverlap=64, Fs=fs, cmap='Blues', vmin=-180);
plt.colorbar(label='dB')
plt.xlabel('$t$ / s')
plt.ylabel('$f$ / Hz')

# plot mirror image sources
plt.figure()
plt.scatter(*xs.T, source_strength*20)
plt.plot(x_mic[:, 0], x_mic[:, 1], 'g.')
plt.gca().add_patch(Rectangle((0, 0), L[0], L[1], fill=False))
plt.xlabel('x / m')
plt.ylabel('y / m')
plt.axis('equal')
plt.title('xy-plane')

plt.figure()
plt.scatter(xs[:, 0], xs[:, 2], source_strength*20)
plt.plot(x_mic[:, 0], x_mic[:, 2], 'g.')
plt.gca().add_patch(Rectangle((0, 0), L[0], L[2], fill=False))
plt.xlabel('x / m')
plt.ylabel('z / m')
plt.axis('equal')
plt.title('xz-plane')


""" Computes the impulse responses in a rectangular room using the 
    mirror image sources model
    * real-valued frequency-independent reflection coefficients considered
    * reflection coefficients applied in an octave-band filterbank
    * fractional delay interpolation with the Lagrange polynomial
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
from iear import octave_filterbank, sos_filterbank

# Constants
c = 343
fs = 44100  # sample rate for boadband response

# Room
L = np.array([3.0, 3.7, 2.8])  # room dimensions
coeffs = .8, .8, .6, .6, .7, .7  # wall reflection coefficients
absorb_coeff = np.array([0.50,  # 15.625 Hz
                         0.55,  # 31.25 Hz
                         0.60,  # 62.5 Hz
                         0.65,  # 125 Hz
                         0.70,  # 250 Hz
                         0.75,  # 500 Hz
                         0.80,  # 1 kHz
                         0.85,  # 2 kHz
                         0.90,  # 4 kHz
                         0.99   # 8 kHz
                         ])     # Kuttruff "Acoustics" Table 13.2
                                # Rockwool 30mm thick on concrete with airgap
ref_coeff = np.sqrt(1 - absorb_coeff**2)

# Point source
x0 = np.array([1.45, 1.83, 1.67])  # source position
#signal = ([1, 0, 0], fs)  # signal for broadband response
source_type = 'point'

# 3D Image sources
max_order =4  # maximum order of image sources
xs, wall_count = sfs.util.image_sources_for_box(x0, L, max_order)
source_strength = np.prod(coeffs**wall_count, axis=1)

# Microphone
x_mic = np.array([0.85, 0.89, 1.23])

# Impulse responses
#N = 4800  # FIR filter length
N = int(2**(np.ceil(np.log2(np.max(np.linalg.norm(xs, axis=-1)) / c * fs))))
#N = int(np.max(np.linalg.norm(xs, axis=-1)) / c * fs)

# Filterbank
f0 = 15.625
N_band = len(absorb_coeff)
imp = np.zeros(N)
imp[0] = 1
filters, frequencies = octave_filterbank(fs, f0, bands=N_band, 
                                              fraction=1, order=4)
subband = sos_filterbank(imp, filters)

# TODO: pre-computation of the DFT of the subband signals

h1 = np.zeros((2*N-1))
for ii, xi in enumerate(xs):
    waveform, shift, offset = impulse_response(xi, x_mic, source_type, fs)
    htemp, _, _ = construct_ir_matrix(waveform, shift, N)
    htemp = np.squeeze(htemp)
    if np.sum(wall_count[ii, :]) != 0:
        reflection = np.sum(subband * ref_coeff ** np.sum(wall_count[ii, :]), axis=-1)
        htemp = fftconvolve(htemp, reflection)
    else:
        hdirect = htemp
        htemp = np.concatenate((htemp, np.zeros((N-1))), axis=-1)
    h1 += htemp


h2 = np.zeros(N)
for ii, xi in enumerate(xs):
    waveform, shift, _ = impulse_response(xi, x_mic, source_type, fs)
    waveform = np.squeeze(waveform)
    htemp = np.zeros(N)
    if np.sum(wall_count[ii, :]) != 0:
        reflection = sos_filterbank(waveform, filters) * ref_coeff**np.sum(wall_count[ii, :], axis=-1)
        superposition = np.sum(reflection, axis=-1)
        htemp[shift[0]:shift[0]+len(waveform)] = superposition
    else:
        htemp[shift[0]:shift[0]+len(waveform)] = waveform
    h2 += htemp
h = h2

# Listening example
s, _ = sf.read('50.flac')  # SQAM male speech
s = s[:3*fs, 0]
y = fftconvolve(s, h / 2 / np.linalg.norm(h))

# Plots
N_ir = len(h)
time = np.arange(N_ir) / fs * 1000
Nf = N_ir // 2 + 1
freq = np.arange(Nf) * fs / (Nf)

# Randomly selected IR - linear scale
plt.figure()
plt.plot(time, h)
plt.xlabel('$t$ / ms')
#plt.ylim(-1, 1)

# Randomly selected IR - dB
plt.figure()
plt.plot(time, db(h))
plt.xlabel('$t$ / ms')
plt.ylabel('Amplitude / dB')
plt.ylim(-120, 0)

# Frequency response
plt.figure()
plt.semilogx(freq, db(np.fft.rfft(h)))
plt.ylim(-60, 0)
plt.xlabel('$f$ / Hz')
plt.ylabel('Magnitude / dB')

# Spectrogram
plt.figure()
plt.specgram(h, NFFT=512, noverlap=256, Fs=fs, cmap='Blues', vmin=-180);
plt.colorbar(label='dB')
plt.xlabel('$t$ / s')
plt.ylabel('$f$ / Hz')

# plot mirror image sources
plt.figure()
plt.scatter(*xs.T, source_strength*20)
plt.plot(x_mic[0], x_mic[1], 'g.')
plt.gca().add_patch(Rectangle((0, 0), L[0], L[1], fill=False))
plt.xlabel('x / m')
plt.ylabel('y / m')
plt.axis('equal')
plt.title('xy-plane')

plt.figure()
plt.scatter(xs[:, 0], xs[:, 2], source_strength*20)
plt.plot(x_mic[0], x_mic[2], 'g.')
plt.gca().add_patch(Rectangle((0, 0), L[0], L[2], fill=False))
plt.xlabel('x / m')
plt.ylabel('z / m')
plt.axis('equal')
plt.title('xz-plane')


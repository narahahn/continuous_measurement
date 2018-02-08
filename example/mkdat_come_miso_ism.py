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
out_dir = '/home/nara/Documents/git/continuous_measurement/example/data_eusipco2018'


# Simulation parameters
M = 1
ism_order = 2

# Constants
c = 343
fs = 16000

# Room
dimension = 5.8, 5.0, 3.0  # room dimensions
coeff = .5, .5, .5, .5, .5, .5  # wall reflection coefficients
room_center = np.array([2.9, 2.5, 1.5])

# Point sources (loudspeaker array)
loudspeaker_array = np.genfromtxt('./data_eusipco2018/university_rostock.csv', delimiter=',')
#idx = 56, 61, 2, 7
idx = 7, 2, 61, 56
idx = idx[:M]
#M = len(idx)
loudspeaker_array = loudspeaker_array[idx, :3] + room_center
source_type = 'point'

# 3D Image sources
#ism_order = 1  # maximum order of image sources
image_sources = []
source_strength = []
for m in range(M):
    x0 = loudspeaker_array[m]
    xs, wall_count = image_sources_for_box(x0, dimension, ism_order)
    image_sources.append(xs)
    source_strength.append(np.prod(coeff**wall_count, axis=1))

# Receiver
array_center = room_center
R = 0.2  # radius

# Moving microphone
Omega = 2 * np.pi / 16
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

filename = 'come_N{}_miso_M{}_ism_order{}_Omega_{}'.format(N, M, ism_order, 
                  int(np.round(np.rad2deg(Omega))))

np.savez('{}/{}.npz'.format(out_dir, filename),
         c, fs, M,
         dimension,
         loudspeaker_array, idx,
         image_sources, source_strength,
         array_center, R, phi,
         Nir, N, p,
         Omega_al, Omega,
         snr,
         s)
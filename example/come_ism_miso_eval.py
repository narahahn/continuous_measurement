"""
Continuous measurement of room impulse responses using a moving microphone.
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


N = 800
M = 1
ism_order = 1
Omega = 2 * np.pi / 16

filename = 'come_N{}_miso_M{}_ism_order{}_Omega_{}'.format(N, M, ism_order, 
                  int(np.round(np.rad2deg(Omega))))

data = np.load('{}/{}.npz'.format(out_dir, filename))
c = data['c']
fs = data['fs']
loudspeaker_array = data['loudspeaker_array']
dimension = data['dimension']
source_strength = data['source_strength']
image_sources = data['image_sources']
room_center = data['room_center']
N = data['N']
Nir = data['Nir']
M = data['M']
R = data['R']
phi = data['phi']
Omega = data['Omega']
Omega_al = data['Omega_al']
s = data['s']
p = data['p']
snr = data['snr']


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


# System identification
int_order = 15
int_type = 'lagrange'
int_type = 'cht'
hhat = system_identification(phi, s, phi_target, p,
                             interpolation=int_type, int_order=int_order)
hhat = np.concatenate([hhat[np.newaxis, :, i * Nir: (i+1) * Nir] for i in range(M)], axis=0)

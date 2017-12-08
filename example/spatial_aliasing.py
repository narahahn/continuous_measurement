"""
Spatial aliasing in continuous measurements

* point source in a free-field
* omnidirectional microphone moving on a circle at a constant speed
* captured signal computed by using fractional delay filters + oversampling
* system identification based on spatial interpolation of a given order
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from sys import path
path.append('../')
from utils import *
from source import *

# Constants
c = 343
fs = 16000

# Source
xs = [0, 2, 0]  # Point source
source_type = 'point'

# Excitation
N = 1600  # excitation period
p = perfect_sequence_randomphase(N)
#p = perfect_sweep(N)

# Receiver
R = 0.5
#Omega = 2 * np.pi / 24
angular_speed = 2* np.pi / np.arange(5, 24, 3)

# Experimental parameters
K = 360  # number of target angles
#Lf = 21  # fractional delay filter length
int_order = 20  # spatial interpolation order
Omega_al = c / N / R  # anti-aliasing angular speed

# Desired impulse responses at selected angles
phi_k = np.linspace(0, 2 * np.pi, num=K, endpoint=False)
x_k = [R*np.cos(phi_k), R*np.sin(phi_k), np.zeros_like(phi_k)]
delay_k, weight_k = greens_point(xs, x_k)
waveform_k, shift_k, offset_k = impulse_response(xs, x_k, 'point', fs)
h0, _, _ = construct_ir_matrix(waveform_k, shift_k, N)


h = np.zeros(np.append(h0.shape, len(angular_speed)))
for ii, Omega in enumerate(angular_speed):
    L = int(2 * np.pi / Omega * fs)
    t = (1/fs) * np.arange(L)
    phi0 = -1e0
    phi = Omega * t + phi0
    xm = [R*np.cos(phi), R*np.sin(phi), np.zeros_like(phi)]

    # Captured signal
    waveform_l, shift_l, offset_l = impulse_response(xs, xm, 'point', fs)
    s = captured_signal(waveform_l, shift_l, p)
    snr = -120
    s += additive_noise(s, snr)

    # System identification
    h[:, :, ii] = system_identification(phi, s, phi_k, p, interpolation='lagrange', int_order=int_order)

nmse = (np.sum((h-h0[:,:,np.newaxis])**2, axis=1) / np.sum(h0**2, axis=1)[:, np.newaxis])**0.5
ave_nmse = np.mean(nmse, axis=0)


# Plots

# Fig. Normalized system distance in dB
plt.figure()
plt.plot(np.rad2deg(phi_k), db(nmse))
plt.xlim(0, 360)
#plt.ylim(-120, 0)
plt.xlabel(r'$\phi$ / $^\circ$')
plt.ylabel('NMSE / dB')
plt.legend(np.round(np.rad2deg(angular_speed)).astype(int), loc='best', title='$\Omega$')
plt.title('Normalized Mean Square Error (order: {})'.format(int_order))

# Fig. Average system distance versus angular speed
plt.figure()
plt.plot(np.rad2deg(angular_speed), db(ave_nmse))
plt.plot([np.rad2deg(Omega_al), np.rad2deg(Omega_al)], [-120, 0], '--')
plt.xlabel(r'$\Omega$ / rad/s')
plt.ylabel('System distance / dB')
plt.xlim(np.min(np.rad2deg(angular_speed)), np.max(np.rad2deg(angular_speed)))
plt.ylim(-120, 0)
plt.title('Mean System Distance (order: {})'.format(int_order))




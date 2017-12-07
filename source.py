"""Compute impulse responses of source models."""
import numpy as np
from scipy.signal import remez, fftconvolve
from sys import path
path.append('../')
from utils import *


def greens_point(xs, x, c=343):
    """Greens function of a point source.

    Parameters
    ----------
    xs : (3,) array_like
        Source position in the Cartesian coordiate (m)
    x : (N, 3) array_like
        Receiver positions in the Cartesian coordinate(m)
    c : float
        Speed of sound (m/s)

    Returns
    -------
    delay : (N,) array_like
        Propagation delay (s)
    weights : (N,) array_like
        Attenuation (1)

    """
    xs = np.array(xs)
    x = np.array(x)
    if x.shape[1] != 3 & x.shape[0] == 3:
        x = x.T
    distance = np.linalg.norm(x - xs[np.newaxis, :], axis=-1)
    return distance/c, 1/4/np.pi/distance


def impulse_response(xs, x, sourcetype, fs, oversample=2, c=343):
    """Impulse responses for a given source type.

    Parameters
    ----------
    xs : (3,) array_like
        Source position in the Cartesian coordinate [m]
    x : (N, 3) array_like
        Receiver positions in the Cartesian coordinate [m]
    sourcetype : string
        Source type e.g. 'point'
    fs : int
        Sampling frequency [Hz]
    oversample : int, optional
        Oversampling factor
    c : float, optional
        Speed of sound

    Returns
    -------
    waveform : (N, C) array_like
        Waveforms (nonzero coefficients) of the impulse resposnes
    shift : (N,) int, array_like
        Shift (number of preceeding zeros) [sample]
    offset : (N,) int, array_like
        Position of the main peak [sample]

    """
    delay, weight = greens_point(xs, x, c)
    waveform_up, shift_up, offset_up = fractional_delay(delay, Lf=23, fs=oversample*fs, type='fast_lagr')

    f = fir_minmax(fs, oversample)
    filtorder = int((len(f)+1)/2)
    waveform_up = fftconvolve(waveform_up, f[np.newaxis, :])

    shift_up -= filtorder
    offset_up -= filtorder
    shift = shift_up // oversample
    res = shift_up % oversample
    offset = offset_up // oversample

    waveform_up = np.column_stack((waveform_up, np.zeros((len(waveform_up), oversample-1))))
    for n in range(len(shift)):
        waveform_up[n, :] = np.roll(waveform_up[n, :], res[n])
    waveform = waveform_up[:, ::oversample]
    return waveform * weight[:, np.newaxis], shift, offset


def fir_minmax(fs, Q):
    """Low-pass filter for sampling rate conversion."""
    fpass = 0.8 * fs / 2
    fstop = 1.0 * fs / 2
    att = -130
    filtorder = Q * 40
    if Q != 1:
        f = remez(2*filtorder+1, [0, fpass, fstop, Q*fs/2], [1, 10**((att)/20)], weight=[1, 10e5], fs=Q*fs)
    else:
        f = np.array(1)
    return f

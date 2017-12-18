"""
Sound field reconstruction based on spherical harmonics representations.

* compute the sound field of a monochromatic plane wave at discrete points
* perform the spherical harmoincs transform of a given order
* reconstruct the sound field on the spherical surface
"""

import numpy as np
import matplotlib.pyplot as plt
import micarray

N = 20  # order of modal beamformer/microphone array
azi_pw = np.pi  # incidence angle of plane wave
pw_angle = (np.pi, np.pi/3)
f = 2000  # frequency
c = 343  # speed of sound
k = 2 * np.pi * f / c  # wave number
r = 0.25  # radius of array

def dot_product_sph(v, u):
    """Evaluate dot-product between u and v in spherical coordinates."""
    return (np.cos(v[0])*np.sin(v[1])*np.cos(u[0])*np.sin(u[1]) +
            np.sin(v[0])*np.sin(v[1])*np.sin(u[0])*np.sin(u[1]) +
            np.cos(v[1])*np.cos(u[1]))

# get quadrature grid (microphone positions) of order N
azi0, elev0, weights = micarray.modal.angular.grid_gauss(N)
Y_p = micarray.modal.angular.sht_matrix(N, azi0, elev0, weights)

# compute microphone signals for an incident broad-band plane wave
p0 = np.exp(1j * k*r * dot_product_sph((azi0, elev0), pw_angle))

# spherical harmonics transform
pnm = np.matmul(np.conj(Y_p), p0)

# reconstruction (synthesis) on the sphere surfrace
Q = 120
azi = np.linspace(0, 2*np.pi, num=Q)
elev = np.linspace(np.pi/Q, np.pi, num=Q)
Azi, Elev = np.meshgrid(azi, elev)
Y = micarray.modal.angular.sht_matrix(N, np.ndarray.flatten(Azi), np.ndarray.flatten(Elev))
phat = np.squeeze(np.matmul(Y.T, pnm[:, np.newaxis]))
phat = np.reshape(phat, (Q, Q))

pd = np.exp(1j * k*r * dot_product_sph((Azi, Elev), pw_angle))

# plots
plt.figure(figsize=(10, 5))
plt.pcolormesh(azi, elev, np.real(pd))
plt.colorbar()
plt.clim(-1, 1)
plt.xlabel('azimuth / rad')
plt.ylabel('elevation / rad')
plt.title('Desired Sound Field (real part)')

plt.figure(figsize=(10, 5))
plt.pcolormesh(azi, elev, np.real(phat))
plt.colorbar()
plt.clim(-1, 1)
plt.xlabel('azimuth / rad')
plt.ylabel('elevation / rad')
plt.title('Reconstructed Sound Field (real part)')

plt.figure(figsize=(10, 5))
plt.pcolormesh(azi, elev, db(phat-pd))
plt.colorbar(label='dB')
plt.clim(-120, 0)
plt.xlabel('azimuth / rad')
plt.ylabel('elevation / rad')
plt.title('Reconstruction Error')

#from matplotlib import cm, colors
#from mpl_toolkits.mplot3d import Axes3D
#x = r * np.sin(Elev) * np.cos(Azi)
#y = r * np.sin(Elev) * np.sin(Azi)
#z = r * np.cos(Elev)

#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.seismic(db(phat-pd)), color='lightgray')
#
#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.seismic(np.real(pd)), cmap='Blues')
#
#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.seismic(np.real(phat)), norm='False')




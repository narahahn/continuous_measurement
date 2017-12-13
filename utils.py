import numpy as np
import scipy.signal as sig
from scipy.special import factorial, comb
import copy


def perfect_sweep(N):
    """
    generate_PerfectSweep returns a periodic perfect sweep

    Parametrs
    ---------
    N :     float
            length of the perfect sequence / sample

    Returns

    p :     array
            perfect_sweep
                
    """
    
    m = np.arange(0, np.floor(N/2+1))
    P_half = np.exp(-1j * 2 * np.pi / N * m**2)

    if (N % 2) == 0:
        P = np.concatenate([P_half, np.conj(np.flipud(P_half[1:-1]))])
    elif (N % 2) == 1:
        P = np.concatenate([P_half, np.conj(np.flipud(P_half[1::]))])
    else:
        print('Invalid length: N.')

    return np.real(np.fft.ifft(P))

def perfect_sequence_randomphase(N):
    """
    Parametrs
    ---------
    N :     int
            length of the perfect sequence / sample

    Returns

    p :     array
            perfect_sweep
                
    """
    
    m = np.arange(0, np.ceil(N/2+1))
    phase = 2 * np.pi * np.random.random(len(m))
#    phase = phase - phase[0]
    phase[0] = 0

    P_half = np.exp(-1j * phase)
    
    if (N % 2) == 0:
        P_half[-1] = 1

    return np.fft.irfft(P_half, n=N)

def twoband_perfect_sweep(N, n_cutoff):
    Nfft = int(np.ceil(N/2+1))
    m = np.arange(0, Nfft)
    P_half = np.exp(-1j * 2 * np.pi / N * m**2)
    
    if (N % 2) == 0:
        P = np.concatenate([P_half, np.conj(np.flipud(P_half[1:-1]))])
    elif (N % 2) == 1:
        P = np.concatenate([P_half, np.conj(np.flipud(P_half[1::]))])
    else:
        print('Invalid length: N.')

    P_low = copy.deepcopy(P)    
    
    P_low = copy.deepcopy(P)
    P_low[n_cutoff:N-n_cutoff+1] = 0
    p_low = np.real(np.fft.ifft(P_low))

    P_high = copy.deepcopy(P)
    P_high[0:n_cutoff] = 0
    P_high[N-n_cutoff+1::] = 0
    p_high = np.real(np.fft.ifft(P_high))

    return np.real(np.fft.ifft(P)), P, p_low, P_low, p_high, P_high


def twoband_perfect_sequence_randomphase(N, n_cutoff):
    Nf = int(np.ceil(N/2+1))
    m = np.arange(0, Nf)
    phase = 2 * np.pi * np.random.rand(len(m))
    phase[0] = 0
    P_full = np.exp(-1j * phase)
    
    if (N % 2) == 0:
        P_full[-1] = 1

#    w, _ = mk_win(2*d+1)
#    w_low = np.concatenate((np.ones(n_cutoff-d-1), w, np.zeros(Nfft-(n_cutoff+d))))
#    w_high = 1 - w_low
        
#     P_low = copy.deepcopy(P_full)
#     P_high = copy.deepcopy(P_full)
    
#     P_low[n_cutoff:] = 0
#     P_high[:n_cutoff] = 0
    P_low = P_full * (m <= n_cutoff)
    P_high = P_full * (m > n_cutoff)
    P_high[::2] = 0

    return np.fft.irfft(P_full, n=N), np.fft.irfft(P_low, n=N), np.fft.irfft(P_high, n=N)

def cconv(x, y, N=None):
    N = np.max((len(x), len(y)))
    return np.fft.irfft( np.fft.rfft(x, n=N) * np.fft.rfft(y, n=N), n=N)

def cxcorr(x, y, N=None):
    N = np.max((len(x), len(y)))
    return np.fft.irfft(np.fft.rfft(x, n=N) * np.fft.rfft(np.roll(y[::-1],1),n=N), n=N)

def time_reverse(x):
    N = len(x)
    return np.roll(x,-1)[N-1::-1]

def db(x, power=False):
    """Convert *x* to decibel.
    Parameters
    ----------
    x : array_like
        Input data.  Values of 0 lead to negative infinity.
    power : bool, optional
        If ``power=False`` (the default), *x* is squared before
        conversion.
    """
    with np.errstate(divide='ignore'):
        return 10 if power else 20 * np.log10(np.abs(x))

def mk_win(L):
    wleft = 0.5*(np.cos(np.pi * np.arange(L) / (L-1))+1)
    wright = wleft[::-1]
    return wleft, wright


def lagr_poly(xi, x):
    """Lagrange polynomail of order n
    
    Parameters
    ----------
    xi : array
         Sequences
    x  : scalar
         input
    
    Returns
    -------
    h : array
        Lagrange polynomial
        
    Notes
    -----
    
    """
    N = len(xi)
    h = np.zeros(N)
    for n in range(N):
        h[n] = np.prod((x-np.delete(xi, n)) / (xi[n]-np.delete(xi, n)))
    return h

def lagr_poly_barycentric(xi, x):
    """Lagrange polynomial of order n using the second Barycentric form
    
    Parameters
    ----------
    xi : array_like
    x : array_like
    
    Returns
    -------
    h : array_like
        Lagrange polynomail
    """
    N = len(xi)
    h = np.zeros(N)
    for n in range(N):
        h[n] = 1 / np.prod(xi[n] - np.delete(xi, n))
    h *= 1 / (x - xi)
    h /= np.sum(h)
    return h

def lagr_poly_barycentric2(xi, x):
    """Lagrange polynomial of order n using the second Barycentric form
    
    Parameters
    ----------
    xi : array_like
    x : array_like
    
    Returns
    -------
    h : array_like
        Lagrange polynomail
    """
    N = len(xi)
    h = np.zeros(N)
    ii = np.arange(N)
    w = comb(N-1, ii) * (-1)**ii
    h = w / (x - xi)
    h /= np.sum(h)
    return h

    

def barycentric_lagrint(ti, yi, t):
    """
    Lagrange interpoltion using the second Barycentric form
    
    Parameters
    ----------
    ti : array_like
        time of the input signals
    yi : array_like
        signal
    t : array_like
        time of the desired signal

    Return
    ------
    y : array_like
        
    """
    if int(order) < 0:
        raise ValueError('Order must be an non-negative integer.')

    N = order + 1
    y = np.zeros_like(t)
    if N % 2 == 0:
        Nhalf = int(N/2)
        for k, tk in enumerate(t):
            n0 = np.searchsorted(ti, tk)
            idx = np.arange(n0-Nhalf, n0+Nhalf)

            w = np.zeros(N)
            for l in range(N):
                w[l] = 1 / np.prod(ti[l] - np.delete(ti, l))
            w *=  1 / (tk-ti[idx])
            w /= np.sum(w)

            y[k] = np.dot(w, yi[idx])
    elif N % 2==1:
        Nhalf = int((N-1)/2)
        for k, tk in enumerate(t):
            n0 = np.argmin(np.abs(ti - tk))
            idx = np.arange(n0-Nhalf, n0+Nhalf+1)

            w = np.zeros(N)
            for l in range(N):
                w[l] = 1 / np.prod(ti[l]) - np.delete(ti, l)
            w *= 1 / (tk-ti[idx])
            w /= np.sum(w)
            
            y[k] = np.dot(w, yi[idx])
    return y

def fdfilt_lagr(tau, Lf, fs):
    """
    Parameters
    ----------
    tau : delay / s
    Lf : length of the filter / sample
    fs : sampling rate / Hz
    
    Returns
    -------
    h : (Lf)
        nonzero filter coefficients
    ni : time index of the first element of h
    n0 : time index of the center of h    
    
    """

    d = tau * fs

    if Lf % 2 == 0:
        n0 = np.ceil(d)
        Lh = int(Lf/2)
        idx = np.arange(n0-Lh, n0+Lh).astype(int)
    elif Lf % 2 == 1:
        n0 = np.round(d)
        Lh = int(np.floor(Lf/2))
        idx = np.arange(n0-Lh, n0+Lh+1).astype(int)
    else:
        print('Invalid value of Lf. Must be an integer')
    return lagr_poly_barycentric2(idx, d), idx[0], n0

def fdfilt_sinc(tau, Lf, fs, beta=8.6):
    """
    Parameters
    ----------
    tau : delay / s
    Lf : length of the filter / sample
    fs : sampling rate / Hz
    
    Returns
    -------
    h : (Lf)
        nonzero filter coefficients
    ni : time index of the first element of h
    n0 : time index of the center of h    
    
    """

    d = tau * fs
    w = np.kaiser(Lf, beta)

    if Lf % 2 == 0:
        n0 = np.ceil(d)
        Lh = int(Lf/2)
        idx = np.arange(n0-Lh, n0+Lh).astype(int)
    elif Lf % 2 == 1:
        n0 = np.round(d)
        Lh = int(np.floor(Lf/2))
        idx = np.arange(n0-Lh, n0+Lh+1).astype(int)
    else:
        print('Invalid value of Lf. Must be an integer')

    return np.sinc(idx - d) * w, idx[0], n0


def fdfilter(xi, yi, x, order, type='lagrange'):
    """
    Lagrange interpolation
    
    Parameters
    ----------
    xi : 
        in accending order
    yi :
    
    x  :
        [xmin, xmax]
    
    Return
    ------
    yi :
    
    """
    N = order+1
    if N%2 == 0:
        Nhalf = N/2
        n0 = np.searchsorted(xi, x)
        idx = np.linspace(n0-Nhalf, n0+Nhalf, num=N, endpoint=False).astype(int)
    elif N%2 == 1:
        Nhalf = (N-1)/2
        n0 = np.argmin(np.abs(xi-x))
        idx = np.linspace(n0-Nhalf, n0+Nhalf+1, num=N, endpoint=False).astype(int)
    else:
        print('order must be an integer')
    
    return np.dot(yi[idx], lagr_poly(xi[idx], x))

def fractional_delay(delay, Lf, fs, type):
    """
    fractional delay filter
    
    Parameters
    ----------
    delay : array
            time-varying delay in sample
    Lf    : int
            length of the fractional delay filter
            
    Returns
    -------
    waveform : array (Lf)
                nonzero coefficients
    shift    : array (Lf)
                indices of the first nonzero coefficient
    offset   : array (Lf)
                indices of the center of the filter
    """
    L = len(delay)
    waveform = np.zeros((L, Lf))
    shift = np.zeros(L)
    offset = np.zeros(L)
    
    if type == 'sinc':
        for n in range(L):
            htemp, ni, n0 = fdfilt_sinc(delay[n], Lf, fs=fs)
            waveform[n, :] = htemp
            shift[n] = ni
            offset[n] = n0
    elif type == 'lagrange':
        for n in range(L):
            htemp, ni, n0 = fdfilt_lagr(delay[n], Lf, fs=fs)
            waveform[n, :] = htemp
            shift[n] = ni
            offset[n] = n0
    elif type == 'fast_lagr':
        d = delay * fs
        if Lf % 2 == 0:
            n0 = np.ceil(d).astype(int)
            Lh = int(Lf/2)
        elif Lf % 2 == 1:
            n0 = np.round(d).astype(int)
            Lh = (np.floor(Lf/2)).astype(int)
        idx_matrix = n0[:, np.newaxis] + np.arange(-Lh, -Lh+Lf)[np.newaxis, :]
        offset = n0
        shift = n0 - Lh
        
        ii = np.arange(Lf)
        common_weight = comb(Lf-1, ii) * (-1)**ii

        is_int = d%1==0
        waveform[~is_int, :] = common_weight[np.newaxis, :] / (d[~is_int, np.newaxis] - idx_matrix[~is_int, :])
        waveform[~is_int, :] /= np.sum(waveform[~is_int, :], axis=-1)[:, np.newaxis]
        waveform[is_int, Lh] = 1
    else:
        print('unknown type')
    return waveform, shift, offset
        

def construct_ir_matrix(waveform, shift, Nh):
    """
    Convert 'waveform' and 'shift' into an IR matrix
    
    Parameters
    ----------
    waveform : array
                nonzero elements of the IRs
    shift :    array
                indices of the first nonzero coefficients
    Nh :       int
                length of each IRs
    
    Returns
    -------
    h : 
        IRs
    H :
        TFs
    Ho :
        CHT spectrum
    
    
    """
    L, Lf = waveform.shape
    h = np.zeros((L, Nh))
    for n in range(L):
        idx = (np.arange(shift[n], shift[n] + Lf)).astype(int)
        h[n, idx] = waveform[n,:]
    H = np.fft.fft(h)
    Ho = (1/L) * np.roll(np.fft.fft(H, axis=0), int(L/2), axis=0)    
    return h, H, Ho


def captured_signal(waveform, shift, p):
    """
    Apply time-varying delay to a perfect sweep
    
    Parameters
    ----------
    waveform : array
                nonzero filter coefficients
    shift :    array
                indices of the first nonzero coefficients
    p :        array
                periodic excitation signal
                
    Returns
    -------
    s : array
                captured signal
    
    """
    return time_varying_delay(waveform, shift, p)
        
def time_varying_delay(waveform, shift, p):
    """
    Apply a time varying delay to an input sequence
    """
    L, Lf = waveform.shape
    N = len(p)
    s = np.zeros(L)
    for n in range(L):
        idx = np.arange(shift[n], shift[n]+Lf).astype(int)
        s[n] = np.dot(p[np.mod(n - idx, N)], waveform[n, :])
    return s


def system_identification(phi, s, phi_target, p, interpolation='lagrange', int_order=1):
    """System identification using spatial interpolation.

    Note: This works only for uniformly moving microphones

    Parameters
    ----------
    phi : (N,) array_like
        Microphone angle [rad]
    s : (N,) array_like
        Captured signal
    phi_target : (K,) array_like
        Target angles [rad]
    p : (L,) array_like
        Excitation signal (one period)
    interpolation : string, optional
        Interpolation method, optioanl
    int_order : int
        Interpolation order

    Return
    ------
    h : (K, N) array_like
        Impulse response coefficients

    """
    L = len(s)
    K = len(phi_target)
    N = len(p)
    h = np.zeros((K, N))
    y = np.zeros((K, N))
    dphi = 2 * np.pi / L * N

    idx_target = (phi_target - phi[0]) / dphi
    L_int = int_order + 1
    idx_int = np.arange(L_int)
    common_weight = comb(int_order, idx_int) * (-1)**idx_int
    for n in range(N):
        if L_int % 2 == 0:
            idx_first = np.ceil(idx_target - n/N).astype(int)
            L_half = int(L_int/2)
        elif L_int % 2 == 1:
            idx_first = np.round(idx_target - n/N).astype(int)
            L_half = int((L_int+1)/2)
        idx = idx_first[:, np.newaxis] + (np.arange(-L_half, -L_half+L_int))[np.newaxis, :]
        offset = idx_first
        shift = offset - L_half
        waveform = common_weight[np.newaxis, :] / (idx_target[:, np.newaxis] - n/N - idx)
        waveform /= np.sum(waveform, axis=-1)[:, np.newaxis]
        s_n = s[n::N]
        is_int = (idx_target-n/N)%1==0
        for k in range(K):
            if is_int[k]:
                y[k, n] = s_n[idx_first[k]]
            else:
                idx_n = np.arange(shift[k], shift[k]+L_int).astype(int)
                y[k, n] = np.dot(s_n[np.mod(idx_n, int(L/N))], waveform[k, :])
    for k in range(K):
        h[k, :] = cxcorr(y[k, :], p)
    return h


def estimate_irs(s, N, idx, order):
    """
    Compute IRs by interpolating the orthogonal coefficients
    
    Paramters
    ---------
    s :
        Captured signal
    N :
        Period of the excitation signal (perfect sweep)
    idx :
        Time indices at which the IRs are computed 
    order :
        Order of the Lagrange interpolator
        
    Returns
    -------
    h : 
        IRs
    """
    L = len(s)
    K = len(idx)
    
    p = perfect_sweep(N)
    prev = np.flipud(np.roll(p, int(N/2-1)))
    
    nn = np.arange(L)
    h = np.zeros((K, N))
    for n in range(K):
        ptemp = np.zeros(N)
        for m in range(N):
            ntemp = nn[m::N]
            sm = s[m::N]
            
            nvect = np.concatenate([ntemp-L, ntemp, ntemp+L])
            pvect = np.concatenate([sm, sm, sm])
            
            ptemp[m] = fdfilter(nvect, pvect, idx[n], order=order)
        h[n, :] = cconv(ptemp, prev)
    return h

def estimate_irs2(s, N, phi, phik, order):
    """
    Compute IRs by interpolating the orthogonal coefficients
    
    Paramters
    ---------
    s :
        Captured signal
    N :
        Period of the excitation signal (perfect sweep)
    phi :
        Anlges where the signal is sampled
    phik :
        Angles where the IRs are computed
    order :
        Order of the Lagrange interpolator
        
    Returns
    -------
    h : 
        IRs
    """
    L = len(s)
    K = len(phik)
    
    p = perfect_sweep(N)
    prev = np.flipud(np.roll(p, int(N/2-1)))
    
    h = np.zeros((K, N))
    for n in range(K):
        ptemp = np.zeros(N)
        for m in range(N):
            phitemp = np.mod(phi[m::N], 2*np.pi)
            sm = s[m::N]
            idx_sort = np.argsort(phitemp)
            phitemp = phitemp[idx_sort]
            sm = sm[idx_sort]
            
            # todo: sort phitemp and sm
            
            phivect = np.concatenate([phitemp-2*np.pi, phitemp, phitemp+2*np.pi])
            pvect = np.concatenate([sm, sm, sm])
            
            ptemp[m] = fdfilter(phivect, pvect, phik[n], order=order)
        h[n, :] = cconv(ptemp, prev)
    return h


def fir_linph_ls(fpass, fstop, att, order, fs, density=10):
    gain_stop = 10**(att/20)
    length = 2*order + 1

    domega = np.pi / order / density    
    omega_passband = np.arange(0, 2*np.pi*fpass/fs, domega)
    omega_stopband = np.arange(np.pi, 2*np.pi*fstop/fs, -domega)[::-1]
    omega = np.concatenate((omega_passband, omega_stopband))
    
    A = np.zeros((len(omega), order+1))
    A[:,0] = 1
    for m in range(1,order+1):
        A[:,m] = 2*np.cos(m*omega)    
    d = np.concatenate((1*np.ones(len(omega_passband)), gain_stop*np.ones(len(omega_stopband))))
    
    h = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A)), np.transpose(d))
    h = np.concatenate((np.flipud(h[1::]), h))
    
#    z = np.roots(h)
#    h0 = np.poly(z[np.abs(z)<1])
#    h0 *= 1/np.sum(h0)
#    
#    g0 = np.flipud(h0)
#    g1 = h0 * (-1)**(np.arange(len(h0))+1)
#    h1 = np.flipud(g1)
    return h


def additive_noise(s, snr):
    """Additive white noise with a given SNR relative to the input signal"""

    additive_noise = np.random.randn(len(s))
    Es = np.std(s)
    En = np.std(additive_noise)
    return additive_noise / En * Es * 10**(snr/20)
    
    
    
    
    

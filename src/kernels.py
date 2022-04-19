import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import constants

"""
This script holds functions used to compute offline signals, window functions,
and filter kernels, including...
- Sinusoidal and square wave test signals
- Hann and Hamming windows
- Impulse response of a bandpass filter and Hilbert transformer
- Frequency response of a bandpass filter and Hilbert transformer

Also implements the overlap-add method for convolution, more as an exercise
than for practical use, since Numpy will do the same thing but more optimized.

"""

F_S = constants.F_S     # sample rate [Hz]
F_STOP_L = constants.F_STOP_L       # [Hz]
F_PASS_L = constants.F_PASS_L       # [Hz]
F_PASS_R = constants.F_PASS_R       # [Hz]
F_STOP_R = constants.F_STOP_R       # [Hz]
F0_HILBERT = constants.F0_HILBERT   # [Hz]
W_PASS_L = 2*np.pi*F_PASS_L/F_S
W_PASS_R = 2*np.pi*F_PASS_R/F_S
M = constants.M


def get_sinusoids(f_list, A_list, phase_list, f_s=constants.F_S, num_periods=5):
    """
    Generates a linear combination of sinusoids evaluated over the specified
    number of signal periods. Used to generate static (offline) test signals.

    Parameters
    ----------
    f_list : ndarray
        1D array of frequencies measured in [Hz]
    A_list : ndarray
        1D array of dimensionless amplitudes
    phase_list : ndarray
        1D array of phases measured in [radians]

    Returns
    -------
    x : ndarray
        Weighted sum of sinusoids with given frequencies, amplitudes and phases.
    t : ndarray
        Time array on which `x` is evaluated
    """
    f0 = np.min(f_list)  # fundamental frequency  # [Hz]
    t0 = 1/f0  # signal period  # [s]
    N0 = f_s * t0  # samples per period

    t_start = 0  # signal start time [s]
    t_end = num_periods*t0  # signal end time [s]
    t = np.linspace(t_start, t_end, int(N0*num_periods))

    x = np.zeros(int(N0*num_periods))  # preallocate array to hold signal
    for i, f in enumerate(f_list):
        x += A_list[i] * np.sin(2*np.pi*f*t + phase_list[i])
    return t, x


def get_square_wave(N, f0, num_periods=5):
    """
    Returns an `N`-term Fourier series approximation of a square wave.

    :param N: number of terms in approximation
    :param f0: fundamental frequency
    """
    n = np.arange(1, 2*N + 1, 2)
    A = np.ones(N)/n
    f = n*f0*np.ones(N)
    phase = np.zeros(N)
    t, x = get_sinusoids(f, A, phase, num_periods=num_periods)
    return t, x


def get_hann(n, M=M):
    """
    Outputs a Hann window of length `M` evaluated on the indices in `n`.

    :param n: 1D numpy array of discrete time indices on which to evaluate h
    :param M: number of coefficients in window (assumed to be odd)
    """
    # Just a vectorized way to return Hann for n \in (0, M-1) and 0 otherwise
    return np.where(np.logical_and(n>=-(M-1)/2, n<=(M-1)/2), 0.5*(1 + np.cos(2*n*np.pi/(M - 1))), 0)


def get_hamming(n, M=M):
    """
    Outputs a Hamming window of length `M` evaluated on the indices in `n`

    :param n: 1D numpy array of discrete time indices on which to evaluate h
    :param M: number of coefficients in window (assumed to be odd)
    """
    # Just a vectorized way to return Hamming for n \in (0, M-1) and 0 otherwise
    return np.where(np.logical_and(n>=-(M-1)/2, n<=(M-1)/2), 0.54 + 0.46 * np.cos(2*n*np.pi/(M - 1)), 0)


def get_h_bp_ideal(n, w1=W_PASS_L, w2=W_PASS_R):
    """
    :param n: 1D numpy array of discrete time indices on which to evaluate h
    :param w1: lower passband frequency
    :param w2: upper passband frequency

    Output impulse response h[n] of ideal bandpass filter
    constructed as a difference of windowed-sinc lowpass filters, both centered at n = 0
    """
    # the special case sinc(0) = 1.0 is already implemented in np.sinc
    # in principle this could be handled "by hand" with sin(n)/n and an if statement returning 1.0 if n == 0.
    return (w2/np.pi)*np.sinc(w2*n/np.pi) - (w1/np.pi)*np.sinc(w1*n/np.pi)


def get_h_bp(M=M, window=True):
    """
    :return: length M 1D numpy array holding bandpass filter's impulse response
    """

    # using n \in (-(M-1)/2, (M-1)/2) (instead of n \in [0, M-1]) permits
    #  the use of unshifted window and sinc functions
    n = np.arange(-(M-1)/2, (M-1)/2 + 1, 1)
    h = get_h_bp_ideal(n)

    if window:
        return get_hann(n)*h
        # return get_hamming(n)*h
    else:
        return h


def get_h_hilbert(M=M, f0=F0_HILBERT, window=True):
    """
    Returns impulse response of a Hilbert transformer centered at n = 0
    :param M: number of samples to use for the transformer
    :param f0: Hilbert transformer will act on frequencies   
               with abs(f) < f0. 
               Set f0 = F_S/2 to act on the entire frequency range.
    :return: length M + 1 1D numpy array holding Hilbert transformer's impulse response
    """
    n = np.arange(-(M-1)/2, (M-1)/2 + 1, 1)
    h = np.divide((np.cos(2*np.pi*n*f0/F_S) - 1), np.pi*n, out=np.zeros(len(n)), where=n!=0)

    if window:
        return get_hann(n)*h
        # return get_hamming(n)*h
    else:
        return h
    

def get_h(window=True):
    """
    Helper function to return combined impulse response of Hilbert transformer and band pass filter
    """
    h_bp = get_h_bp(window=window)
    h_hilbert = get_h_hilbert(window=window)
    h_total = np.convolve(h_bp, h_hilbert)[int((M-1)/2):-int((M-1)/2)]
    return h_total


def get_H_bp():
    """
    Returns:
     - H: complex-valued 1D numpy array holding filter frequency response
     - f: real-valued 1D numpy array holding frequency values on which H is defined
    """
    h = get_h_bp()

    N = 4096  # number of points in frequency axis
    f = np.linspace(-F_S/2, F_S/2, N) # frequency axis
    H = fft(h, N)  # frequency response
    H = fftshift(H)
    return f[1:], H[1:]  # hack to remove first point with -/infty dB


def get_H_hilbert():
    """
    Returns:
     - H: complex-valued 1D numpy array holding Hilbert transformer's frequency response
     - f: real-valued 1D numpy array holding frequency values on which H is defined
    """
    h = get_h_hilbert()

    N = 4096  # number of points in frequency axis
    f = np.linspace(-F_S/2, F_S/2, N) # frequency axis
    H = fft(h, N)  # frequency response
    H = fftshift(H)
    return f, H


def get_H(window=True):
    """
    Helper function to return combined frequency response of Hilbert transformer and band pass filter
     - H: complex-valued 1D numpy array holding filter's frequency response
     - f: real-valued 1D numpy array holding frequency values on which H is defined
    """
    h = get_h(window=window)

    N = 4096  # number of points in frequency axis
    f = np.linspace(-F_S/2, F_S/2, N) # frequency axis
    H = fft(h, N)  # frequency response
    H = fftshift(H)
    return f, H


def convolve_oa(h, x):
    """
    Uses the overlap-add method to convole impulse resposne h with signal x

    :param h: impulse response of length M
    :param x: signal of arbitrary length, probably N >> M
    """
    # assume M = 329
    # which motivates signal buffer size L = 185
    #  ... so buffer-impulse response convolution has length L + M - 1 = 512
    # or L = 697 and convolution length 1024

    N = len(x)
    M = len(h)
    L = 697
    N_chunks = int(N/L)
    N_remainder = int(N%L)

    buf = np.zeros(L)
    y = np.zeros(len(h) + len(x) - 1)  # full convolution of h and x

    for n in range(N_chunks):
        # buffer n-th chunk of input x
        # convolve buffer and impulse response; len(conv) = M+L-1
        # place result of convolution in n-th chunk of output y
        # ... and in the first M-1 elements of the (n + 1)-th chunk of y
        # append last M-1 elements of previous convolution...
        # ...to first M-1 elements of current convolution
        # save last M-1 elements of current convolution
        buf = x[n*L:(n+1)*L]
        conv = np.convolve(h, buf)
        print("Length buf: {}".format(len(buf)))
        print("Length conv: {}".format(len(conv)))
        y[n*L:n*L + (L + M - 1)] += conv

    # corner case of left-over elements in input
    if N_remainder == 0:
        return y
    else:
        # initialize zero-filled buffer of length L 
        # fill first samples of buffer with remaining samples at the end of input signal
        buf = x[-N_remainder:]
        conv = np.convolve(h, buf)
        y[N_chunks*L:] += conv
        return y


def practice():
    n = np.arange(-(M-1)/2, (M-1)/2 + 1, 1)
    print(n)
    out = np.where


if __name__ == "__main__":
    # get_h_bp()
    # get_h()
    practice()

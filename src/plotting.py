import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import constants, kernels

""" 
This script generates the figures used in report accompanying the project.
Plots are generated in PDF format for integration with `pdflatex`, but the
figure format can easily be changed (to e.g. JPEG or PNG) by changed by
modifying the `format` parameter passed to Matplotlib's `savefig` function.
"""

F_S = constants.F_S            # sample rate [Hz]
NYQUIST = constants.NYQUIST    # Nyqustion rate [Hz]
F_STOP_L = constants.F_STOP_L  # [Hz]
F_PASS_L = constants.F_PASS_L  # [Hz]
F_PASS_R = constants.F_PASS_R  # [Hz]
F_STOP_R = constants.F_STOP_R  # [Hz]

color_in = constants.color_blue
color_out = constants.color_orange_dark

save_figs = True
fig_dir = "../media/"


def plot_h(h):
    """
    Generic function to plot impulse response
    :param h: real 1D numpy array holding filter's impulse response
    """
    plt.plot(h, marker='.', linestyle='-')
    plt.show()


def plot_h_bp():
    """
    Plots the bandpass filter's impulse response
    """
    h = kernels.get_h_bp()
    h = h/np.max(abs(h))
    plt.plot(h, marker='.', linestyle='-')
    plt.xlabel("Digital index $ n $")
    plt.ylabel("Normalized amplitude $ h[n] $")
    plt.title("Bandpass Filter's Impulse Response")
    plt.show()


def plot_h_hilbert():
    """
    Plots the Hilbert transform's impulse response
    """
    h = kernels.get_h_hilbert()
    h = h/np.max(abs(h))
    plt.plot(h, marker='.', linestyle='-')
    plt.xlabel("Digital index $ n $")
    plt.ylabel("Normalized amplitude $ h[n] $")
    plt.title("Hilbert Transformer's Impulse Response")
    plt.show()


def plot_h_total():
    """
    Plots the complete filter's impulse response
    """
    h = kernels.get_h()
    h = h/np.max(abs(h))
    plt.plot(h, marker='.', linestyle='-')
    plt.xlabel("Digital index $ n $")
    plt.ylabel("Normalized amplitude $ h[n] $")
    plt.title("Complete Filter's Impulse Response")
    plt.show()


def plot_h_all():
    """
    Plots the bandpass, Hilbert, and convolved BP-Hilbert filters' impulse
    response on three separate Matplotlib axes.
    """
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(7, 3))

    # Bandpass
    ax = axes[0]
    h = kernels.get_h_bp()
    h = h/np.max(abs(h))
    ax.plot(h, marker='.', linestyle='-', linewidth=1, color=color_blue)
    ax.set_xlabel("Digital index $ n $")
    ax.set_ylabel("Normalized amplitude $ h[n] $")
    ax.set_title("Bandpass $ h_{\mathrm{bp}} $")

    # Hilbert transformer
    ax = axes[1]
    h = kernels.get_h_hilbert()
    h = h/np.max(abs(h))
    ax.plot(h, marker='.', linestyle='-', linewidth=1, color=color_blue)
    ax.set_xlabel("Digital index $ n $")
    ax.set_title("Hilbert $ h_{\mathrm{H}} $")

    # Both bandpass and Hilbert transformer
    ax = axes[2]
    h = kernels.get_h()
    h = h/np.max(abs(h))
    ax.plot(h, marker='.', linestyle='-', linewidth=1, color=color_orange_dark)
    ax.set_xlabel("Digital index $ n $")
    ax.set_title("$ h_{\mathrm{bp}} * h_{\mathrm{H}} $")

    plt.tight_layout()

    if save_figs:
        plt.savefig(fig_dir + "h-all.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()


def mark_bandpass_specs(ax, line_color=color_blue, fill_color="#CCCCCC"):
    """
    Helper method plots lines and shades forbidden regions
     to show the bandpass filter's specifications.
    To avoid repeating the same code in three functions.
    """
    xmin, xmax = 0, 3000
    ymin, ymax = -70, 2
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.fill_between(np.linspace(1000, 2000), -1, -40, fc=fill_color)
    ax.fill_between(np.linspace(0, 500), -1, -40, fc=fill_color)
    ax.fill_between(np.linspace(2500, 3000), -1, -40, fc=fill_color)
    ax.fill_between(np.linspace(0, 3000), 1, ymax, fc=fill_color)

    ax.hlines(1, 0, 3000, linestyle='--', color=line_color)
    ax.hlines(-1, 0, 3000, linestyle='--', color=line_color)
    ax.hlines(-40, 0, 3000, linestyle='--', color=line_color)
    

def plot_H_hilbert_abs():
    """
    Plots the Hilbert transformer's frequency response
    """
    f, H = kernels.get_H_hilbert()
    xmin, xmax = 0, 6000
    ymin, ymax = -70, 2
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    fill_color = "#AAAAAA"
    plt.fill_between(np.linspace(1000, 2000), -1, -40, fc=fill_color)
    plt.fill_between(np.linspace(0, 500), -1, -40, fc=fill_color)
    plt.fill_between(np.linspace(2500, 3000), -1, -40, fc=fill_color)
    plt.fill_between(np.linspace(0, 3000), 1, ymax, fc=fill_color)

    plt.hlines(1, 0, 3000, linestyle='--')
    plt.hlines(-1, 0, 3000, linestyle='--')
    plt.hlines(-40, 0, 3000, linestyle='--')

    plt.xlabel("Frequency $ f $ [Hz]")
    plt.ylabel("Attenuation $ |H\, | $ [dB]")
    H_dB = 20 * np.log10(np.abs(H))
    plt.plot(f, H_dB, marker='.')
    plt.show()


def plot_H_total_abs():
    """
    Plots the combined Hilbert-bandpass filter's frequency response
    """
    f, H = kernels.get_H()
    fig, ax = plt.subplots(figsize=(7, 4))
    mark_bandpass_specs(ax)
    ax.set_xlabel("Frequency $ f $ [Hz]")
    ax.set_ylabel("Attenuation $ |H\, | $ [dB]")
    ax.set_title("Windowed Frequency Response")
    H_dB = 20 * np.log10(np.abs(H))
    ax.plot(f, H_dB, marker='.', markersize=3)
    plt.tight_layout()
    plt.show()


def plot_H_total_abs_close_up():
    """
    Plots the combined Hilbert-bandpass filter's frequency response, zooming in
    on the passband and adding annotations to show the frequency response meets
    all required filter specifications.

    """
    f, H = kernels.get_H()
    fig, ax = plt.subplots(figsize=(7, 4))
    mark_bandpass_specs(ax)
    ax.set_xlabel("Frequency $ f $ [Hz]")
    ax.set_ylabel("Attenuation $ |H\, | $ [dB]")
    ax.set_title("Windowed Frequency Response")
    ax.set_xlim(450, 2550)
    ax.set_ylim(-42, 2)
    H_dB = 20 * np.log10(np.abs(H))
    ax.plot(f, H_dB, markersize=3, linewidth=2)
    plt.tight_layout()
    if save_figs:
        plt.savefig(fig_dir + "H-abs-close.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()


def plot_H_total_abs_no_window():
    """
    Plots the combined Hilbert-bandpass filter's frequency response with an unwindowed impulse resposne.
    """
    fig, ax = plt.subplots()
    f, H = kernels.get_H(window=False)
    mark_bandpass_specs(ax)
    ax.set_xlabel("Frequency $ f $ [Hz]")
    ax.set_ylabel("Attenuation $ |H\, | $ [dB]")
    ax.set_title("Unwindowed Frequency Response")
    H_dB = 20 * np.log10(np.abs(H))
    ax.plot(f, H_dB, marker='.')
    plt.tight_layout()
    plt.show()


def plot_H_compare_window():
    """
    Compares the windowed and unwindowed frequency response of the combined
    Hilbert-bandpass filter.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

    # WINDOWED
    ax = axes[0]
    f, H = kernels.get_H()
    # mark_bandpass_specs(ax)

    xmin, xmax = 0, 3000
    ymin, ymax = -70, 5
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Frequency $ f $ [Hz]")
    ax.set_ylabel("Attenuation $ |H\, | $ [dB]")
    ax.set_title("Windowed")
    H_dB = 20 * np.log10(np.abs(H))
    ax.plot(f, H_dB, color=color_blue)

    # UNWINDOWED
    ax = axes[1]
    f, H = kernels.get_H(window=False)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Frequency $ f $ [Hz]")
    ax.set_ylabel("Attenuation $ |H\, | $ [dB]")
    ax.set_title("Un-windowed")
    H_dB = 20 * np.log10(np.abs(H))
    ax.plot(f, H_dB, color=color_orange_dark)
    plt.tight_layout()

    if save_figs:
        plt.savefig(fig_dir + "H-abs.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()


def plot_H_bp_angle():
    """
    Plots the filter's frequency response
    :param f: real 1D numpy array holding frequency axis data points in Hz
    :param H: complex 1D numpy array holding filter's frequency response
    """
    f, H = kernels.get_H_bp()
    fig, ax = plt.subplots()
    mark_angle_axes(ax)
    ax.plot(f, np.unwrap(np.angle(H)), marker='.')
    ax.set_title("Bandpass Filter's Phase Angle")
    plt.show()


def plot_H_hilbert_angle():
    """
    Plots the filter's frequency response
    :param f: real 1D numpy array holding frequency axis data points in Hz
    :param H: complex 1D numpy array holding filter's frequency response
    """
    f, H = kernels.get_H_hilbert()
    fig, ax = plt.subplots()
    mark_angle_axes(ax)
    ax.plot(f, np.unwrap(np.angle(H)), marker='.')
    ax.set_title("Hilbert Transformer's Phase Angle")
    plt.show()


def plot_H_total_angle():
    """
    Plots the complete filter's unwrapped phase response.

    """
    f, H = kernels.get_H()
    fig, ax = plt.subplots(figsize=(7,4))
    mark_angle_axes(ax)
    ax.plot(f, np.unwrap(np.angle(H)))

    ax.set_title("Filter's Phase Response")
    plt.tight_layout()

    if save_figs:
        plt.savefig(fig_dir + "H-angle.pdf", bbox_inches='tight', pad_inches=0)

    plt.show()


def mark_angle_axes(ax):
    """
    Helper method plots lines and mark axes of phase angle plots
    To avoid repeating the same code in three functions.
    """
    xmin, xmax = -1.333*F_STOP_R, 1.333*F_STOP_R
    ax.set_xlim(xmin, xmax)
    # ymin, ymax = -80, 2
    # plt.ylim(ymin, ymax)
    ax.set_xlabel("Frequency $ f $ [Hz]" )
    ax.set_ylabel(r"Unwrapped phase angle $\theta$")


def test_filter_time_domain_shift():
    """
    Demonstrates the filter's phase shift in the time domain using a sinusoidal
    signal in the filter's passband.
    """
    f_s=constants.F_S
    f = 1500
    frequencies = np.array([f])
    amplitudes = np.ones(len(frequencies))
    phase = np.zeros(len(frequencies))
    num_periods=15

    t, x_in = kernels.get_sinusoids(frequencies, amplitudes, phase, num_periods=num_periods)

    phase = (np.pi/2) * np.ones(len(frequencies))
    _, x_shifted = kernels.get_sinusoids(frequencies, amplitudes, phase, num_periods=num_periods)

    h = kernels.get_h()
    M = len(h)
    # xf = np.convolve(h, x_in)[int((M-1)/2):-int((M-1)/2)]
    xf = kernels.convolve_oa(h, x_in)[int((M-1)/2):-int((M-1)/2)]

    # Show five periods from period 5 to period 10 (Dropping first and last few
    # periods to avoid transient effects)
    T_start = 5
    T_end = 10
    N_start = int(T_start * f_s / f)
    N_end = int(T_end * f_s / f)

    x_in = x_in[N_start:N_end]
    x_shifted = x_shifted[N_start:N_end]
    xf = xf[N_start:N_end]

    plt.subplots(figsize=(7, 4))
    plt.xlabel("Digital index $ n $")
    plt.ylabel("Amplitude [AU]")

    plt.plot(x_in, color=color_in, label='input ({} Hz)'.format(f), linewidth=1.5)
    plt.plot(x_shifted, color=color_in, linestyle='--',
            marker='o',
            label='input shifted by $\pi/2$\n(for reference)')
    plt.plot(xf, color=color_out, label='filtered', linewidth=1.5)

    plt.legend(loc='lower left', framealpha=0.95)
    plt.tight_layout()

    if save_figs:
        plt.savefig(fig_dir + "test-time-shift.pdf")

    plt.show()


def test_filter_square():
    """
    Tests the filter on square wave input signal and shows the filtered output
    in both the time and frequency domains. The time domain representation
    shows the filter's Hilbert transformer (phase shift) and bandpass (reducing
    a square wave to its sinusoidal fundamental) components.

    """
    h = kernels.get_h()
    M = len(h)

    f0 = 1200  # fundamental frequency
    n_terms = 10
    n_terms = min(n_terms,  int(NYQUIST/f0))
    t, x_in = kernels.get_square_wave(n_terms, f0, num_periods=4)
    N = len(x_in)

    x_out = np.convolve(h, x_in)[int((M-1)/2):-int((M-1)/2)]
    f = np.linspace(-F_S/2, F_S/2, N)
    X_in = fft(x_in, n=N)
    X_in = fftshift(X_in)
    X_out = fft(x_out, n=N)
    X_out = fftshift(X_out)
    X_max = max(np.abs(X_in))  # normalization factor

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.5, 4))

    # TIME DOMAIN
    ax = axes[0]
    ax.set_xlabel("Digital index $ n $")
    ax.set_ylabel("Normalize amplitude")
    ax.plot(x_in, label='input', color=color_in, linestyle=':', linewidth=1.5)
    ax.plot(x_out, label='filtered', color=color_out, linewidth=1.5)

    ax.legend(loc='lower left', framealpha=0.95)

    # FREQUENCY DOMAIN
    ax=axes[1]
    xmin, xmax = 0, 17500
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Frequency $ f $ [Hz]" )

    # stem plot parameters
    marker_in = '.'
    marker_out = 'd'
    m_size_in = 10
    m_size_out = 7
    l_width = 1.5

    # Input stem plot
    (markers, stemlines, baseline) = ax.stem(f, np.abs(X_in)/X_max, label='input')
    plt.setp(markers, marker=marker_in, markerfacecolor=color_in, markeredgecolor="none", markersize=m_size_in)  # markers
    plt.setp(baseline, linestyle="-", color=color_in)  # baseline
    plt.setp(stemlines, linestyle=":", color=color_in, linewidth=l_width)  # stemlines

    # Output stem plot
    (markers, stemlines, baseline) = ax.stem(f, np.abs(X_out)/X_max, label='filtered')
    plt.setp(markers, marker=marker_out, markerfacecolor=color_out, markeredgecolor="none", markersize=m_size_out)  # markers
    plt.setp(baseline, linestyle="-", color=color_out)  # baseline
    plt.setp(stemlines, linestyle="-", color=color_out, linewidth=l_width)  # stemlines

    ax.legend()
    plt.tight_layout()

    if save_figs:
        plt.savefig(fig_dir + "test-square.pdf")

    plt.show()


def test_filter_bandpass():
    """
    Tests the filter on superposition of sinusuoids and shows the filtered
    output in both the time and frequency domains. This plot is meant to fully
    show the filter's bandpass properties---the filter strips the sinusoidal
    components that lie outside the passband and passes only the component in
    the passband

    """
    h = kernels.get_h()
    M = len(h)

    frequencies = np.array([300, 500, 1500, 2500, 2700])
    amplitudes = np.ones(len(frequencies))
    phase = np.zeros(len(frequencies))
    t, x_in = kernels.get_sinusoids(frequencies, amplitudes, phase, num_periods=3)

    N = len(x_in)

    x_out = np.convolve(h, x_in)[int((M-1)/2):-int((M-1)/2)]
    f = np.linspace(-F_S/2, F_S/2, N)
    X_in = fft(x_in, n=N)
    X_in = fftshift(X_in)
    X_out = fft(x_out, n=N)
    X_out = fftshift(X_out)
    X_max = max(np.abs(X_in))  # normalization factor

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8.5, 4))

    # TIME DOMAIN
    ax = axes[0]
    ax.set_xlabel("Digital index $ n $")
    ax.set_ylabel("Normalize amplitude")
    ax.plot(x_in, label='input', color=color_in, linestyle=':', linewidth=1)
    ax.plot(x_out, label='filtered', color=color_out, linewidth=1.5)

    ax.legend(loc='lower left', framealpha=0.95)

    # FREQUENCY DOMAIN
    ax=axes[1]
    xmin, xmax = 0, 3000
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Frequency $ f $ [Hz]" )

    # stem plot parameters
    marker_in = '.'
    marker_out = 'd'
    m_size_in = 10
    m_size_out = 7
    l_width = 1.5

    # Input stem plot
    (markers, stemlines, baseline) = ax.stem(f, np.abs(X_in)/X_max, label='input')
    plt.setp(markers, marker=marker_in, markerfacecolor=color_in, markeredgecolor="none", markersize=m_size_in)  # markers
    plt.setp(baseline, linestyle="-", color=color_in)  # baseline
    plt.setp(stemlines, linestyle=":", color=color_in, linewidth=l_width)  # stemlines

    # Output stem plot
    (markers, stemlines, baseline) = ax.stem(f, np.abs(X_out)/X_max, label='filtered')
    plt.setp(markers, marker=marker_out, markerfacecolor=color_out, markeredgecolor="none", markersize=m_size_out)  # markers
    plt.setp(baseline, linestyle="-", color=color_out)  # baseline
    plt.setp(stemlines, linestyle="-", color=color_out, linewidth=l_width)  # stemlines

    ax.legend(loc='lower left', framealpha=0.95)

    plt.tight_layout()

    if save_figs:
        plt.savefig(fig_dir + "test-bandpass.pdf")

    plt.show()


if __name__ == "__main__":
    plot_h_all()
    plot_H_compare_window()
    plot_H_total_abs_close_up()
    plot_H_total_angle()

    test_filter_time_domain_shift()
    test_filter_square()
    test_filter_bandpass()


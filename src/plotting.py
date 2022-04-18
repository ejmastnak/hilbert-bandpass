import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import constants, bandpass

F_S = constants.F_S  # sample rate [Hz]
NYQUIST = constants.NYQUIST    # Nyqustion rate [Hz]
F_STOP_L = constants.F_STOP_L  # [Hz]
F_PASS_L = constants.F_PASS_L  # [Hz]
F_PASS_R = constants.F_PASS_R  # [Hz]
F_STOP_R = constants.F_STOP_R  # [Hz]


# start matplotlib defaults configuration
# ------------------------------------------ #
import matplotlib as mpl
import matplotlib.font_manager as font_manager
mpl.rcParams['font.family'] = 'serif'

try:
    cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
    mpl.rcParams['font.serif'] = cmfont.get_name()
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['axes.unicode_minus'] = False  # so the minus sign '-' displays correctly in plots
except FileNotFoundError as error:
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'STIXGeneral'

plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('axes', titlesize=16)    # fontsize of titles
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['lines.linewidth'] = 1
# ------------------------------------------ #
# end atplotlib defaults configuration

color_blue = "#244d90"         # darker teal / blue
color_orange_dark = "#91331f"  # dark orange
color_orange_mid = "#e1692e"  # mid orange

color_in = color_blue
color_out = color_orange_dark

save_figs = True
fig_dir = "../figures/"


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
    h = bandpass.get_h_bp()
    h = h/np.max(abs(h))
    # plt.figure(figsize=(4, 4))
    plt.plot(h, marker='.', linestyle='-')
    plt.xlabel("Digital index $ n $")
    plt.ylabel("Normalized amplitude $ h[n] $")
    plt.title("Bandpass Filter's Impulse Response")
    plt.show()


def plot_h_hilbert():
    """
    Plots the Hilbert transform's impulse response
    """
    # h = bandpass.get_h_hilbert(f0=constants.NYQUIST)
    h = bandpass.get_h_hilbert()
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
    h = bandpass.get_h()
    h = h/np.max(abs(h))
    plt.plot(h, marker='.', linestyle='-')
    plt.xlabel("Digital index $ n $")
    plt.ylabel("Normalized amplitude $ h[n] $")
    plt.title("Complete Filter's Impulse Response")
    plt.show()


def plot_h_all():
    """
    Plots the bandpass, Hilbert, and convolved BP-Hilbert filters' impulse response
     on three separate axes
    """
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(7, 3))

    # bandpass
    ax = axes[0]
    h = bandpass.get_h_bp()
    h = h/np.max(abs(h))
    ax.plot(h, marker='.', linestyle='-', linewidth=1, color=color_blue)
    ax.set_xlabel("Digital index $ n $")
    ax.set_ylabel("Normalized amplitude $ h[n] $")
    ax.set_title("Bandpass $ h_{\mathrm{bp}} $")

    # Hilbert
    ax = axes[1]
    h = bandpass.get_h_hilbert()
    h = h/np.max(abs(h))
    ax.plot(h, marker='.', linestyle='-', linewidth=1, color=color_blue)
    ax.set_xlabel("Digital index $ n $")
    ax.set_title("Hilbert $ h_{\mathrm{H}} $")

    # both
    ax = axes[2]
    h = bandpass.get_h()
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
    f, H = bandpass.get_H_hilbert()
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
    f, H = bandpass.get_H()
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
    Plots the combined Hilbert-bandpass filter's frequency response
    """
    f, H = bandpass.get_H()
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
    f, H = bandpass.get_H(window=False)
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
    Compares the windowed and unwindowed frequency response of the combined Hilbert-bandpass filter

    Reduntant in that code is copied from plot_H functions above. Oh wel.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

    # WINDOWED
    ax = axes[0]
    f, H = bandpass.get_H()
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
    f, H = bandpass.get_H(window=False)

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
    f, H = bandpass.get_H_bp()
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
    f, H = bandpass.get_H_hilbert()
    fig, ax = plt.subplots()
    mark_angle_axes(ax)
    ax.plot(f, np.unwrap(np.angle(H)), marker='.')
    ax.set_title("Hilbert Transformer's Phase Angle")
    plt.show()


def plot_H_total_angle():
    """
    Plots the complete filter's angle response
    :param f: real 1D numpy array holding frequency axis data points in Hz
    :param H: complex 1D numpy array holding filter's frequency response
    """
    f, H = bandpass.get_H()
    fig, ax = plt.subplots(figsize=(7,4))
    mark_angle_axes(ax)
    ax.plot(f, np.unwrap(np.angle(H)))

    # # arrow
    # ax.annotate("", 
    #         xy=(500, -45), xytext=(0, -20), 
    #         ha="center", va="center",
    #         arrowprops=dict(facecolor='black', width=0.05, headwidth=4, headlength=6, shrink=0.15))
    # # text
    # ax.annotate("500 Hz",
    #         xy=(500, -45), xytext=(0, -20), 
    #         ha="center", va="center",
    #         bbox=dict(facecolor='#FFFFFF', edgecolor='#222222', boxstyle='round,pad=0.2'))

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
    Tests entire filter in time domain
    """
    frequencies = np.array([1000])
    amplitudes = np.ones(len(frequencies))
    phase = np.zeros(len(frequencies))
    num_periods=20

    t, x_in = bandpass.get_sinusoids(frequencies, amplitudes, phase, num_periods=num_periods)

    phase = (np.pi/2) * np.ones(len(frequencies))
    _, x_shifted = bandpass.get_sinusoids(frequencies, amplitudes, phase, num_periods=num_periods)

    h = bandpass.get_h()
    M = len(h)
    # xf = np.convolve(h, x_in)[int((M-1)/2):-int((M-1)/2)]
    xf = bandpass.convolve_oa(h, x_in)[int((M-1)/2):-int((M-1)/2)]

    plt.subplots(figsize=(7, 4))
    plt.xlabel("Digital index $ n $")
    plt.ylabel("Amplitude [AU]")
    plt.plot(x_in, color=color_in, label='input', linewidth=1.5)
    plt.plot(x_shifted, color=color_in, linestyle='--', label='input shifted by $\pi/2$\n(for reference)')
    plt.plot(xf, color=color_out, label='filtered', linewidth=1.5)
    plt.legend(loc='lower left', framealpha=0.95)
    plt.tight_layout()

    # if save_figs:
    #     plt.savefig(fig_dir + "test-time-shift.pdf")

    plt.show()


def test_filter_square():
    """
    Tests bandpass and Hilbert transform component of filter in both time and frequency domain using a square wave
    """
    h = bandpass.get_h()
    M = len(h)

    f0 = 1200  # fundamental frequency
    n_terms = 10
    n_terms = min(n_terms,  int(NYQUIST/f0))
    t, x_in = bandpass.get_square_wave(n_terms, f0, num_periods=4)
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
    Tests bandpass and Hilbert transform component of filter 
     in both time and frequency domain using an input signal
     varying over a wide range of frequencies
    """
    h = bandpass.get_h()
    M = len(h)

    frequencies = np.array([300, 500, 1500, 2500, 2700])
    amplitudes = np.ones(len(frequencies))
    phase = np.zeros(len(frequencies))
    t, x_in = bandpass.get_sinusoids(frequencies, amplitudes, phase, num_periods=3)

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
    # plot_h_bp()
    # plot_h_hilbert()
    # plot_h_total()
    # plot_h_all()

    # plot_H_hilbert_abs()
    # plot_H_total_abs()
    # plot_H_total_abs_no_window()
    # plot_H_compare_window()
    plot_H_total_abs_close_up()

    # plot_H_hilbert_angle()
    # plot_H_total_angle()

    # test_filter_time_domain_shift()
    # test_filter_square()
    # test_filter_bandpass()




import numpy as np
import pyaudio
import matplotlib
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
import bandpass, constants

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

plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('axes', titlesize=16)    # fontsize of titles
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
# ------------------------------------------ #
# end atplotlib defaults configuration


class AudioStream(object):
    def __init__(self):
        # change plotting modes
        self.PLOT_WAVE = 0
        self.PLOT_SPECTRUM = 1
        self.GRAPH_MODE = self.PLOT_WAVE
        # self.GRAPH_MODE = self.PLOT_SPECTRUM

        # get impulse response
        self.h = bandpass.get_h()
        self.h = self.h.astype('float32')
        self.M = len(self.h)
        self.h_delay = np.zeros(self.M, dtype=np.float32)
        self.h_delay[int((self.M - 1)/2)] = 1.0

        # initialize audio steam constants
        self.CHUNK = 2*4096 - self.M + 1  # so FFT when convolving buffer and impulse response is a power of two
        # note: anything much below 4096 produces frequent buffer overflows when reading from stream
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100  # [Hz]
        self.NYQUIST = self.RATE/2
        self.N_FFT = 2*self.CHUNK  # number of points to use for FFTs

        # configure maximum frequency to plot in spectrum
        fraction_of_nyquist_to_show = 4
        self.F_MAX = self.NYQUIST/fraction_of_nyquist_to_show
        self.F_MAX_INDEX = int(0.5*self.N_FFT/fraction_of_nyquist_to_show)

        # initialize/preallocate arrays 
        self.f = np.linspace(-self.NYQUIST, self.NYQUIST, self.N_FFT)
        self.f_truncated = np.linspace(0, self.F_MAX, self.F_MAX_INDEX)
        self.audio_data_float32 = np.zeros(self.CHUNK, dtype=np.float32)
        self.audio_spectrum = np.zeros(self.F_MAX_INDEX, dtype=np.float32)
        self.conv_prev = np.zeros(self.M - 1)
        self.conv_prev_delay = np.zeros(self.M - 1)

        # some plotting parameters
        color_blue = "#244d90"         # darker teal / blue
        color_teal = "#3997bf"         # lighter teal/blue
        color_orange_dark = "#91331f"  # dark orange
        color_orange_mid = "#e1692e"   # mid orange
        self.color_in = color_teal
        self.color_out = color_orange_dark
        self.width_in = 1.5
        self.width_out = 2

        if self.GRAPH_MODE == self.PLOT_WAVE:
            self.initialize_waveform_graph()
        else:
            self.initialize_spectrum_graph()
        self.start_stream()


    def initialize_waveform_graph(self):
        """
        Initialize matplotlib graph to plot input and filtered audio waveform
        """
        matplotlib.use('TkAgg')
        self.fig, ax = plt.subplots(nrows=1, ncols=1)

        # input signal
        # ax.set_ylim(-0.5, 0.5)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlim(0, 2048)
        ax.set_xlabel("Buffer index")
        ax.set_ylabel("Amplitude (32-bit float)")
        self.line_wave_raw, = ax.plot(np.arange(0, self.CHUNK, 1), self.audio_data_float32, color=self.color_in, linestyle='--', lw=self.width_in)
        self.line_wave_filt, = ax.plot(np.arange(0, self.CHUNK, 1), self.audio_data_float32, color=self.color_out, lw=self.width_out)

        plt.show(block=False)


    def initialize_spectrum_graph(self):
        """
        Initialize matplotlib graph to plot input and filtered audio spectrum
        """
        matplotlib.use('TkAgg')
        self.fig, ax = plt.subplots(nrows=1, ncols=1)

        # spectrum ylim is emperically set
        y_max = 750

        # input signal
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude (float32, AU)")
        self.line_spectrum_raw, = ax.plot(self.f_truncated, self.audio_spectrum, color=self.color_in, linestyle='--', linewidth=self.width_in)
        self.line_spectrum_filt, = ax.plot(self.f_truncated, self.audio_spectrum, color=self.color_out, lw=self.width_out)

        plt.show(block=False)


    def start_stream(self):
        """
        Initializes and starts audio stream.
        Begins reading and plotting audio data.
        """
        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK)

        self.stream.start_stream()

        while True:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                # data = self.stream.read(self.CHUNK)
                self.audio_data_float32 = np.frombuffer(data, dtype=np.float32)

                if self.GRAPH_MODE == self.PLOT_WAVE:
                    # self.update_wave_graph()
                    self.update_wave_graph()
                else:
                    self.update_spectrum_graph()

            except KeyboardInterrupt:
                self.stream.stop_stream()
                self.stream.close()
                self.p.terminate()
                print("Exiting")
                

    def update_wave_graph(self):
        """
        Test with delayed input
        """
        data_delayed, audio_filtered = self.get_filtered_chunk(self.audio_data_float32)

        self.line_wave_raw.set_ydata(data_delayed)
        self.line_wave_filt.set_ydata(audio_filtered)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        

    def update_spectrum_graph(self):
        """
        Computes spectrum of current buffer and updates graph
        """

        data_delayed, audio_filtered = self.get_filtered_chunk(self.audio_data_float32)

        spectrum_raw = fft(data_delayed, n=self.N_FFT)
        spectrum_raw = spectrum_raw[:self.F_MAX_INDEX]

        spectrum_filtered = fft(audio_filtered, n=self.N_FFT)
        spectrum_filtered = spectrum_filtered[:self.F_MAX_INDEX]

        self.line_spectrum_raw.set_ydata(np.abs(spectrum_raw))
        self.line_spectrum_filt.set_ydata(np.abs(spectrum_filtered))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        

    def get_filtered_chunk(self, data_in):
        """
        Test with delaying input as well as output
        """
        L = len(data_in)
        conv = np.convolve(self.h, data_in)
        data_out = conv[:L]
        # appends last M-1 elements of previous conv to first M-1 elements of current conv
        data_out[:self.M - 1] += self.conv_prev
        # saves last M-1 elements of current conv
        self.conv_prev = conv[-(self.M - 1):]

        # for delaying input signal
        conv_delay = np.convolve(self.h_delay, data_in)
        data_in_delayed = conv_delay[:L]
        data_in_delayed[:self.M - 1] += self.conv_prev_delay
        self.conv_prev_delay = conv_delay[-(self.M - 1):]
        return data_in_delayed, data_out


    def close_stream(self):
        """
        Closes audio stream
        """
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("Stream closed.")

    def test_output(self):
        """
        Initializes and starts audio stream.
        Prints byte-string and 16 output to the console
        """
        self.p = pyaudio.PyAudio()

        self.CHUNK = 10
        self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK)

        self.stream.start_stream()

        while True:
            try:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                self.audio_data_float32 = np.frombuffer(data, dtype=np.float32)

                print("Start iteration")
                print(data)
                print(self.audio_data_float32)
                print("End iteration")

            except KeyboardInterrupt:
                self.stream.stop_stream()
                self.stream.close()
                self.p.terminate()
                print("Exiting")


if __name__ == "__main__":
    AudioStream()

import numpy as np
import pyaudio
import matplotlib
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
import bandpass, constants

class AudioStream(object):
    def __init__(self):

        # Plotting modes to plot either time-domain waveform or
        # frequency-domain spectrum.
        self.PLOT_WAVE = 0
        self.PLOT_SPECTRUM = 1
        self.GRAPH_MODE = self.PLOT_WAVE

        # Get filter's impulse response from `bandpass.py`
        self.h = bandpass.get_h()
        self.h = self.h.astype('float32')
        self.M = len(self.h)
        self.h_delay_by_M = np.zeros(self.M, dtype=np.float32)
        self.h_delay_by_M[int((self.M - 1)/2)] = 1.0

        # Initialize audio steam constants
        # --------------------------------------------- #
        # A buffer (`CHUNK`) size much below 4096 produces frequent buffer
        # overflows when reading from stream. The `- self.M + 1` line in the
        # buffer size declaration ensures the FFT used to convolve the buffer
        # and impulse response acts on a signal with a power of two elements.
        self.CHUNK = 4096 - self.M + 1  
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = constants.F_S
        self.NYQUIST = self.RATE/2
        self.N_FFT = 2*self.CHUNK  # number of points to use for FFTs
        # --------------------------------------------- #

        # Configure maximum frequency to plot in spectrum
        fraction_of_nyquist_to_show = 4
        self.F_MAX = self.NYQUIST/fraction_of_nyquist_to_show
        self.F_MAX_INDEX = int(0.5*self.N_FFT/fraction_of_nyquist_to_show)

        # Initialize/preallocate arrays 
        self.f = np.linspace(-self.NYQUIST, self.NYQUIST, self.N_FFT)
        self.f_truncated = np.linspace(0, self.F_MAX, self.F_MAX_INDEX)
        self.audio_data_float32 = np.zeros(self.CHUNK, dtype=np.float32)
        self.audio_spectrum = np.zeros(self.F_MAX_INDEX, dtype=np.float32)
        self.filter_conv_prev = np.zeros(self.M - 1)
        self.delay_conv_prev = np.zeros(self.M - 1)

        # Some plotting parameters
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
        Initialize matplotlib graph to plot input and filtered audio waveform.
        """
        matplotlib.use('TkAgg')
        self.fig, ax = plt.subplots(nrows=1, ncols=1)

        # To accomodate full y axis range of audio waveform. Consider shrinking
        # to e.g. (-0.5, 0.5) for a more "zoomed-in" version.
        ax.set_ylim(-1.0, 1.0)

        # Show only first 1024 frames of audio buffer for a more "zoomed-in"
        # view of the input and filtered waveform.
        ax.set_xlim(0, 1024)  

        ax.set_xlabel("Buffer index")
        ax.set_ylabel("Amplitude (32-bit float)")
        self.line_wave_raw, = ax.plot(np.arange(0, self.CHUNK, 1),
                self.audio_data_float32, color=self.color_in, linestyle='--',
                lw=self.width_in)
        self.line_wave_filt, = ax.plot(np.arange(0, self.CHUNK, 1),
                self.audio_data_float32, color=self.color_out,
                lw=self.width_out)

        plt.show(block=False)


    def initialize_spectrum_graph(self):
        """
        Initialize matplotlib graph to plot input and filtered audio spectrum
        """
        matplotlib.use('TkAgg')
        self.fig, ax = plt.subplots(nrows=1, ncols=1)

        # This value is emperically chosen to accomodate the spectrum's entire
        # y axis range. May need some tweaking on different hardware.
        y_max = 750

        # input signal
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude (float32, AU)")
        self.line_spectrum_raw, = ax.plot(self.f_truncated,
                self.audio_spectrum, color=self.color_in, linestyle='--',
                linewidth=self.width_in)
        self.line_spectrum_filt, = ax.plot(self.f_truncated,
                self.audio_spectrum, color=self.color_out, lw=self.width_out)

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
                # Note `exception_on_overflow=False` to let program continue
                # without crashing in case of buffer overflow.
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                self.audio_data_float32 = np.frombuffer(data, dtype=np.float32)

                if self.GRAPH_MODE == self.PLOT_WAVE:
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
        Computes the next filtered audio buffer and accordingly updates the
        data displayed on the waveform graph.
        """
        data_delayed, audio_filtered = self.get_filtered_buffer(self.audio_data_float32)

        self.line_wave_raw.set_ydata(data_delayed)
        self.line_wave_filt.set_ydata(audio_filtered)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        

    def update_spectrum_graph(self):
        """
        Computes the spectrum of the next filtered audio buffer and accordingly
        updates the data displayed on the spectrum graph.
        """

        buffer_delayed, buffer_filtered = self.get_filtered_buffer(self.audio_data_float32)

        spectrum_raw = fft(buffer_delayed, n=self.N_FFT)
        spectrum_raw = spectrum_raw[:self.F_MAX_INDEX]

        spectrum_filtered = fft(buffer_filtered, n=self.N_FFT)
        spectrum_filtered = spectrum_filtered[:self.F_MAX_INDEX]

        self.line_spectrum_raw.set_ydata(np.abs(spectrum_raw))
        self.line_spectrum_filt.set_ydata(np.abs(spectrum_filtered))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        

    def get_filtered_buffer(self, buffer_in):
        """
        Applies the Hilbert transform filter to the inputted audio buffer.

        Parameters
        ----------
        buffer_in : ndarray
            1D Numpy array holding a buffer of audio data

        Returns
        -------
        buffer_out : ndarray
            Filtered version of `buffer_in`
        buffer_in_delayed : ndarray
            A copy of `buffer_in` delayed by the same number of indices as
            `buffer_out` is delayed relative to `buffer_in`. Loosely, FIR
            filtering necessarily introduces a time delay between input and
            output, and this delay step returns a copy of the input signal that
            is realigned with the filtered output.
        
        """
        L = len(buffer_in)
        filter_conv = np.convolve(self.h, buffer_in)
        buffer_out = filter_conv[:L]

        # Append last M-1 elements of previous convolution to first M-1
        # elements of current convolution
        buffer_out[:self.M - 1] += self.filter_conv_prev

        # Save last M-1 elements of this iteration's convolution for use in the
        # next iteration
        self.filter_conv_prev = filter_conv[-(self.M - 1):]

        # Create a delayed copy of `buffer_in` that aligns with `buffer_out`
        delay_conv = np.convolve(self.h_delay_by_M, buffer_in)
        buffer_in_delayed = delay_conv[:L]
        buffer_in_delayed[:self.M - 1] += self.delay_conv_prev
        self.delay_conv_prev = delay_conv[-(self.M - 1):]

        return buffer_in_delayed, buffer_out


    def close_stream(self):
        """
        Closes audio stream
        """
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        print("Stream closed.")

if __name__ == "__main__":
    AudioStream()

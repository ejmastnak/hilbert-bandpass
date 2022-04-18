import numpy as np
import pyaudio 
from numpy.fft import fft, fftshift
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import sys
import time

# control plot mode
PLOT_WAVE = 0
PLOT_SPECTRUM = 1
PLOT_BOTH = 2
PLOT_MODE = PLOT_BOTH

class WaveformViewer(object):
    def __init__(self):

        self.init_window()
        self.init_audio()

        if PLOT_MODE == PLOT_WAVE:
            self.init_waveform_plot()
        elif PLOT_MODE == PLOT_SPECTRUM:
            self.init_spectrum_plot()
        else:
            self.init_waveform_plot()
            self.init_spectrum_plot(row=2)

    def init_window(self):
        """
        Initializes objects needed for the plot window to appear
        """
        pg.setConfigOptions(antialias=True)
        self.app = QtGui.QApplication(sys.argv)
        self.win = pg.GraphicsWindow(title='Waveform Viewer')
        self.win.setWindowTitle('Waveform and Spectrum Viewer')

    def init_audio(self):
        """ 
        Initializes audio-related constants.
        Creates the PyAudio stream used to read in audio data 
        from the computer microphone.
        """

        # initialize audio steam constants
        self.CHUNK = 2048  # TODO: does 2048 cause buffer overflow?
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100  # sample rate [Hz]
        self.NYQUIST = self.RATE/2

        # the (arbitrary) number of points to use for FFT
        self.N_FFT = 2*self.CHUNK  

        # create stream object
        self.pa = pyaudio.PyAudio()

        self.stream = self.pa.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK)

        self.stream.start_stream()

    def init_waveform_plot(self, row=1, col=1):
        """ 
        Initializes data and x/y range for waveform plot. 

        """
        # create waveform's pyqtgraph PlotItem
        self.waveform_PI = self.win.addPlot(title='Waveform',
                row=row, col=col)

        # create waveform's pyqtgraph PlotDataItem
        self.waveform_PDI = self.waveform_PI.plot(pen='c', width=3)

        # the samples' indices, i.e. 0, 1, 2, ..., CHUNK - 1
        self.samples = np.arange(0, self.CHUNK, 1)

        # set x/y range of waveform plot
        self.waveform_PI.setXRange(0, self.CHUNK)
        self.waveform_PI.setYRange(-1.0, 1.0)  # assuming float-32 audio data

    def init_spectrum_plot(self, row=1, col=1):
        """ Initializes data and x/y range for spectrum plot. """
        # create spectrums's pyqtgraph PlotItem
        self.spectrum_PI = self.win.addPlot(title='Spectrum',
                row=row, col=col)

        # create spectrums's pyqtgraph PlotDataItem
        self.spectrum_PDI = self.spectrum_PI.plot(pen='red', width=3)

        # set maximum frequency to plot in spectrum graph
        # since Nyquist rate of ~22000 Hz is unnecessarily large
        # most interesting frequences occur up to about 8000 Hz
        fraction_of_nyquist_to_plot = 4
        self.f_max = self.NYQUIST/fraction_of_nyquist_to_plot
        self.f_max_index = int(0.5*self.N_FFT/fraction_of_nyquist_to_plot)

        # the problem's native Nyquist frequency range
        self.f = np.linspace(-self.NYQUIST, self.NYQUIST, self.N_FFT)

        # truncated frequencies intended for plotting spectrum
        self.f_truncated = np.linspace(0, self.f_max, self.f_max_index)

        # set x/y range of spectrum plot
        y_max = 50  # spectrum ylim is set emperically
        self.spectrum_PI.setXRange(0, self.f_max)
        self.spectrum_PI.setYRange(0, y_max)

    def start(self):
        """ This makes "the plot show up and stuff work" but how?  """
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def callback(self):
        """
        Callback function for the QTimer loop, used to update audio plot.
        The function:
          - reads in a buffer of data from audio stream (in byte form)
          - converts buffered audio data to float32
          - takes FFT of buffer to get audio spectrum
          - updates waveform plot
          - updates spectrum plot
        """
        waveform = self.stream.read(self.CHUNK,
                exception_on_overflow=False)
        waveform_float32 = np.frombuffer(waveform,
                dtype=np.float32)

        if PLOT_MODE == PLOT_WAVE:
            self.waveform_PDI.setData(self.samples, waveform_float32)
        elif PLOT_MODE == PLOT_SPECTRUM:
            spectrum = fft(waveform_float32, n=self.N_FFT)
            spectrum = spectrum[:self.f_max_index]
            self.spectrum_PDI.setData(self.f_truncated, np.abs(spectrum))
        else:
            spectrum = fft(waveform_float32, n=self.N_FFT)
            spectrum = spectrum[:self.f_max_index]
            self.waveform_PDI.setData(self.samples, waveform_float32)
            self.spectrum_PDI.setData(self.f_truncated, np.abs(spectrum))

    def anim(self):
        """
        TODO: how does QTimer work?
        Then calls self.start()
        """
        timer = QtCore.QTimer()
        timer.timeout.connect(self.callback)
        timer.start(30)
        self.start()

if __name__ == "__main__":
    viewer = WaveformViewer()
    viewer.anim()

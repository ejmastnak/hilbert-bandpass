# Bandpass Hilbert Transformer

Project goal: design and implement a finite impulse response (FIR) band-pass Hilbert transformer that works in near real-time on 44100 Hz audio signals.

Context: This was originally a project for the third-year course *Zajemanje in obdelava podatkov* (Data Acquisition and Processing) at the Faculty of Mathematics and Physics and the University of Ljubljana, Slovenia, which I later put on GitHub for a public audience.

https://user-images.githubusercontent.com/50270681/164107533-03ac9bce-3aa5-489e-8ef5-0c89c68f67d6.mp4

## Project structure

- The `report` directory contains the LaTeX source files for the project report
- The `media` directory holds figures and a video demonstrating real-time filtering.
- The `src` directory contains the project source code.
  Here is a breakdown of what each file does:
  - `constants.py` stores project-wide filter specifications
  - `kernels.py` contains functions for offline computation of test signals, window functions, and filter kernels and frequency responses
  - `plotting.py` generates the figures used in the LaTeX report
  - `realtime.py` is used to test the filter in real time using an audio signal from the computer microphone
  - `hilbert.mplstyle` is a basic style sheet for Matplotlib plots

## Dependencies

The project is programmed in Python 3.
Both the offline component (computing filter kernels and generating figures) and real-time component require:
- [NumPy](https://www.numpy.org): for implementation of the fast Fourier transform and common mathematical functions
- [Matplotlib](https://www.matplotlib.org): for plotting

The real-time component additionally requires:
- [PortAudio](http://www.portaudio.com/): for capturing audio from the computer's microphone
- [PyAudio](http://people.csail.mit.edu/hubert/pyaudio/): provides Python bindings for the C code in PortAudio
- [Tkinter](https://docs.python.org/3/library/tkinter.html): for the real-time plotting interface

To locally compile the LaTeX report you'll need:
- A LaTeX installation (e.g. [TeXLive](https://www.tug.org/texlive/))
- The `pdflatex` and `latexmk` programs, which should be included with most LaTeX installations
- The [`biber`](https://github.com/plk/biber) backend for BibLaTeX (to manage the bibliography)

Warning: the audio capture involved in the real-time filtering component may cause headaches depending on your operating system and hardware.
I got everything working on macOS, which uses the Core Audio API, but ran into troubles with clipping and latency on a Linux machine using PulseAudio.
Your mileage may vary.

## Building

To locally build the figures and LaTeX report:

1. First change into the `src` directory and run `python plotting.py`, which will generate figures and save them to the `media` directory.
1. Once figures are generated, change into the `report` directory and run `latexmk report.tex`.
   You'll probably want to configure `latexmk` to use `pdflatex` as the LaTeX engine for compilation, since the report uses PDF graphics.

## License

All original writing, figures, and video, including the contents of the `report` and `media` directories, are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
The project source code, i.e. the contents of the `src` directory, is licensed under the [MIT License](https://opensource.org/licenses/MIT).

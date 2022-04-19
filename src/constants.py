"""
Sample rate, filter specs, and some plotting colors related to the Hilbert
transformer project.
"""

F_S = 44100         # sample rate [Hz]
NYQUIST = F_S/2     # Nyquist rate [Hz]
PI = 3.141592653589793

# Number of coefficients used in the filter's impulse response. In principle
# ~329 should be enough to meet the filter specifications, but you might want
# to bump this up by a hundred or so for a narrower transition band.
M = 329  

# Stopband and passband frequencies
F_STOP_L = 500      # [Hz]
F_PASS_L = 750     # [Hz]
F_PASS_R = 2200     # [Hz]
F_STOP_R = 2500     # [Hz]

# Frequency up to which Hilbert transformer will act
F0_HILBERT = NYQUIST

# Normalized versions of stopband and passband frequencies
W_STOP_L = 2*PI*F_STOP_L/F_S
W_PASS_L = 2*PI*F_PASS_L/F_S
W_PASS_R = 2*PI*F_PASS_R/F_S
W_STOP_R = 2*PI*F_STOP_R/F_S

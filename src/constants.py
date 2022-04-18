"""
Sample rate and filter specs related to the Hilbert transformer project.
"""

F_S = 41000         # sample rate [Hz]
NYQUIST = F_S/2     # Nyquist rate [Hz]
PI = 3.141592653589793
M = 529  # number of filter coefficients

F_S = 41000  # sample rate [Hz]
F_STOP_L = 500      # [Hz]
F_PASS_L = 750     # [Hz]
F_PASS_R = 2200     # [Hz]
F_STOP_R = 2500     # [Hz]

# F0_HILBERT = 3500   # [Hz]
F0_HILBERT = NYQUIST

# normalized frequencies
W_STOP_L = 2*PI*F_STOP_L/F_S
W_PASS_L = 2*PI*F_PASS_L/F_S
W_PASS_R = 2*PI*F_PASS_R/F_S
W_STOP_R = 2*PI*F_STOP_R/F_S

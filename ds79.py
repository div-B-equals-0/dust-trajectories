"""
Implementation of equation 4, 5, 6 of Draine & Salpeter 1979ApJ...231...77D
"""
import numpy as np

G0_CONSTANT = 3.0*np.sqrt(np.pi)/8.0
def G0(s):
    "Draine & Salpeter's approximate version"
    return (s/G0_CONSTANT)*np.sqrt(1.0 + (G0_CONSTANT*s)**2)

def G2byG0(s):
    "My improved approximation"
    return 1.0/(2.0 + s**2 + s**4)

# Characteristic speed at 1e4 K
CHARACTERISTIC_SPEED = 12.8486
COULOMB_LAMBDA = 20.0

def Fdrag(w, T=1e4, phi=10.0):
    """
    Drag force in units of gas pressure times geometric cross-section:

    F / (2 n k T pi a^2)

    Sum of Stokes and Coulomb terms
    """
    w0 = CHARACTERISTIC_SPEED*np.sqrt(T/1e4)
    s = w / w0
    return G0(s) * (1.0 + COULOMB_LAMBDA*phi**2*G2byG0(s))
        

"""
Implementation of equation 4, 5, 6 of Draine & Salpeter 1979ApJ...231...77D
"""
import numpy as np

G0_CONSTANT = 3.0*np.sqrt(np.pi)/8.0
def _G0(s):
    "Draine & Salpeter's approximate version"
    return (s/G0_CONSTANT)*np.sqrt(1.0 + (G0_CONSTANT*s)**2)

def _G2byG0(s):
    "My improved approximation"
    return 1.0/(2.0 + s**2 + s**4)

# Characteristic speed at 1e4 K
CHARACTERISTIC_SPEED = 12.8486
COULOMB_LAMBDA = 20.0


class Collider(object):
    def __init__(self, name, A=1.0, Z=1.0, abun=1.0):
        self.name = name
        self.A = A
        self.Z = Z
        self.abun = abun

COLLIDERS = [
    Collider("proton"),
    Collider("electron", A=5.446e-4),
    Collider("He+", A=4.0, abun=0.1),
    # Collider("He++", A=4.0, Z=2.0, abun=0.1),
]
        
def Fdrag(w, T=1e4, phi=10.0):
    """
    Sum of Stokes and Coulomb contributions to gas-grain drag as a
    function of relative velocity `w` (in km/s), gas temperature `T`
    (in Kelvin), and grain potential `phi` (in units of kT).  Returns
    drag force in units of gas pressure times geometric cross-section:

    F / (2 n k T pi a^2)

    """
    rslt = np.zeros_like(w)
    for c in COLLIDERS:
        w0 = CHARACTERISTIC_SPEED*np.sqrt(T/1e4/c.A)
        s = w / w0
        rslt += c.abun*_G0(s) * (1.0 + COULOMB_LAMBDA*(c.Z*phi)**2*_G2byG0(s))
    return rslt
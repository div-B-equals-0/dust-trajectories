"""
Implementation of equation 4, 5, 6 of Draine & Salpeter 1979ApJ...231...77D
"""
import numpy as np
from scipy.special import erf

SQRT_PI = np.sqrt(np.pi)
G0_CONSTANT = 3.0 * SQRT_PI / 8.0


def _G0(s):
    "Draine & Salpeter's approximate version"
    return (s / G0_CONSTANT) * np.sqrt(1.0 + (G0_CONSTANT * s) ** 2)


def _G2byG0(s):
    "My improved approximation to the ratio"
    return 1.0 / (2.0 + s ** 2 + s ** 4)


# Exact versions, because why not?
def _G0_exact(s):
    "DS79 equation 5a"
    rslt = (s ** 2 + 1.0 - 1.0 / (4.0 * s ** 2)) * erf(s)
    rslt += (s + 1.0 / (2 * s)) * np.exp(-(s ** 2)) / SQRT_PI
    return rslt


def _G2_exact(s):
    "DS79 equation 6a"
    return erf(s) / s ** 2 - 2 * np.exp(-(s ** 2)) / (s * SQRT_PI)


def _ln_Lambda(n, T):
    """
    Coulomb logarithm: natural log of plasma parameter
    """
    return 23.267 + 1.5 * np.log(T / 1e4) - 0.5 * np.log(n)


# Thermal speed of proton at 1e4 K
CHARACTERISTIC_SPEED = 12.8486
APPROX_COULOMB_LAMBDA = 20.0


class Collider(object):
    """
    Different colliders that can exert drag force on the grains
    """

    def __init__(self, name, A=1.0, Z=1.0, abun=1.0):
        """
        Each collider has atomic mass `A`, charge `Z`, and abundance
        `abun` (by number, relative to protons)
        """
        self.name = name
        self.A = A
        self.Z = Z
        self.abun = abun


DEFAULT_COLLIDERS = [
    Collider("proton"),
    Collider("electron", A=5.446e-4),
    Collider("He+", A=4.0, abun=0.1),
]

# For hard ionizing spectrum, replace He+ with He++
HARD_IONIZATION_COLLIDERS = DEFAULT_COLLIDERS[:-1] + [
    Collider("He++", A=4.0, Z=2.0, abun=0.1),
]


SMALL_NUMBER = 1e-7


def Fdrag(w, T=1e4, phi=10.0, n=1.0, colliders=DEFAULT_COLLIDERS):
    """
    Sum of Epstein and Coulomb contributions to gas-grain drag as a
    function of relative velocity `w` (in km/s), gas temperature `T`
    (in Kelvin), grain potential `phi` (in units of kT), and density
    `n` (in pcc).  Optional argument `colliders` should be a list of
    `Collider` instances.

    Returns drag force in units of gas pressure times geometric
    cross-section:

    F / (2 n k T pi a^2)
    """
    rslt = np.zeros_like(w)
    for c in colliders:
        w0 = CHARACTERISTIC_SPEED * np.sqrt(T / 1e4 / c.A)
        s = w / w0
        # Guard against divide by zero
        s = np.abs(s) + SMALL_NUMBER
        # Can't use += operator when w is a vector
        rslt = rslt + c.abun * (
            _G0_exact(s) + _ln_Lambda(n, T) * (c.Z * phi) ** 2 * _G2_exact(s)
        )
    return rslt


def Fdrag_components(w, T=1e4, phi=10.0, n=1.0, colliders=DEFAULT_COLLIDERS):
    """
    Individual components of gas-grain drag as a function of relative
    velocity `w` (in km/s), gas temperature `T` (in Kelvin), grain
    potential `phi` (in units of kT), and density `n` (in pcc).
    Optional argument `colliders` should be a list of `Collider`
    instances.

    Returns dict of drag forces in units of gas pressure times
    geometric cross-section (see `Fdrag`).  Dict is keyed by the
    collider name ('proton', 'electron', etc) and each item is a
    2-tuple of the (Epstein G0, Coulomb G2) contributions from that
    collider.  If `w` is an array then each element of each 2-tuple is
    an array of the same shape as `w`.
    """
    rslt = {}
    for c in colliders:
        w0 = CHARACTERISTIC_SPEED * np.sqrt(T / 1e4 / c.A)
        s = w / w0
        # return dict of 2-tuples for (Epstein, Coulomb) contributions
        rslt[c.name] = (
            c.abun * _G0_exact(s),
            c.abun * _ln_Lambda(n, T) * (c.Z * phi) ** 2 * _G2_exact(s),
        )
    return rslt


def Fdrag_approx(w, T=1e4, phi=10.0):
    """
    OLD APPROXIMATE VERSION.  Sum of Stokes and Coulomb contributions
    to gas-grain drag as a function of relative velocity `w` (in
    km/s), gas temperature `T` (in Kelvin), and grain potential `phi`
    (in units of kT).  Returns drag force in units of gas pressure
    times geometric cross-section:

    F / (2 n k T pi a^2)
    """
    rslt = np.zeros_like(w)
    for c in DEFAULT_COLLIDERS:
        w0 = CHARACTERISTIC_SPEED * np.sqrt(T / 1e4 / c.A)
        s = w / w0
        rslt = rslt + c.abun * _G0(s) * (
            1.0 + APPROX_COULOMB_LAMBDA * (c.Z * phi) ** 2 * _G2byG0(s)
        )
    return rslt

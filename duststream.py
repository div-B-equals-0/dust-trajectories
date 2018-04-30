import numpy as np
import ds79

MICRON = 1e-4                   # cm

class DustStream(object):
    """A dust stream incident upon a star"""
    def __init__(self, L4=1.0, vinf=40.0, n=1.0,
                 T=1e4, kappa=600.0, Qp=2.0, a=0.02, rho_d=3.0, phi=10.0):
        # Star properties
        self.L4 = L4
        # Plasma stream properties
        self.vinf = vinf
        self.n = n
        self.T = T
        self.kappa = 600.0            # cm^2 / g
        self.cs = 11.4*np.sqrt(T/1e4) # km / s
        self.taustar = 0.0089*(kappa/600.0)*np.sqrt(L4*n)*(10.0/vinf)
        self.Rstar = 2.21*np.sqrt(L4/n)*(10.0/vinf)  # pc
        # Dust grain properties
        self.Qp = Qp
        self.a = a              # micron
        self.rho_d = rho_d      # g / cm^3
        self.kappa_d = 3.0*Qp/(4.0*a*MICRON*rho_d)
        self.drag_constant = (4.0/self.Qp)*(self.cs*self.taustar*self.kappa_d
                                            /(self.vinf*self.kappa))**2
        self.phi = phi
         
def dydt_1d(y, t, s):
    """
    1D equation of motion for dust grain, subject to radiation force
    and drag force.

    Extra argument `s` is a DustStream() instance
    """
    x, u = y
    dxdt = u
    w = (u + 1.0)*s.vinf
    dudt = 0.5*(x**(-2) - s.drag_constant*ds79.Fdrag(w, s.T, s.phi))
    return [dxdt, dudt]


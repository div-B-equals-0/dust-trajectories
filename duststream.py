import numpy as np
from scipy.optimize import brentq
import ds79

MICRON = 1e-4                   # cm

class DustStream(object):
    """A dust stream incident upon a star"""
    def __init__(self, L4=1.0, vinf=40.0, n=1.0, a=0.02,
                 thetaB=0.0,
                 eta=0.01, T=1e4, kappa=600.0, Qp=2.0, rho_d=3.0, phi_norm=1.0):
        # Star properties
        self.L4 = L4
        self.eta = eta
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
        self.phi_norm = phi_norm

        # Magnetic field
        self.thetaB = thetaB    # B-field angle from x-axis, in degrees
        self.b_hat = np.array([
            np.cos(np.deg2rad(thetaB)),
            np.sin(np.deg2rad(thetaB)),
            0.0,
        ])
        
        # R**: drag-free turn-around radius
        self.Rstarstar = 2*(self.kappa_d/self.kappa)*self.taustar*self.Rstar
        # P_rad/P_gas at R**
        self.Upstarstar = (self.vinf/self.cs)**2 * (self.Rstar/self.Rstarstar)**2
        # Bow-shock radius in strong coupling limit
        a, b = 0.0, 2*np.sqrt(1.0 + self.eta)
        x = brentq(xfunc, a, b, args=(self.taustar, self.eta))
        self.R0 = x*self.Rstar

        
def xfunc(x, ts, eta):
    """Function to be zeroed to find R_0 / R_*"""
    return x**2 - (1.0 - np.exp(-2*ts*x)) - eta


def dydt_1d(y, t, s):
    """
    1D equation of motion for dust grain, subject to radiation force
    and drag force.

    Extra argument `s` is a DustStream() instance
    """
    x, u = y
    dxdt = u
    w = (u + 1.0)*s.vinf
    dudt = total_accel(x, w, s)
    return [dxdt, dudt]


def total_accel(x, w, s):
    """
    Total 1d radial acceleration on a grain: radiative + drag

    Input parameters: `x` is radius in units of R_** (drag-free
    turnaround radius).  `w` is relative velocity in km/s.  `s` is a
    `DustStream` object.
    """
    return 0.5*(x**(-2)
                - s.drag_constant*ds79.Fdrag(w, s.T, phi(x, s)))


def dydt_2d(state, t, s):
    """
    2D equation of motion for dust grain, subject to radiation force
    and drag force. Used in ODE evolution of `state` vector with `t` 

    Extra argument `s` is a DustStream() instance
    """
    x, u, y, v = state
    dxdt = u
    dydt = v
    dudt, dvdt = vector_accel_2d(x, y, u, v, s)
    return [dxdt, dudt, dydt, dvdt]


def vector_accel_2d(x, y, u, v, s, ugas=-1.0, vgas=0.0):
    """
    Total 2d vector acceleration on a grain: radiative + drag

    Input parameters: `x`, `y` are position in units of R_**
    (drag-free turnaround radius).  `u`, `v` are velocity in units of
    v_inf.  `s` is a `DustStream` object.  Optional arguments `ugas`,
    `vgas` are for if we do the back reaction on the gas.

    Returns: `ax`, `ay`, the cartesian components of acceleration
    """
    # Radius from star
    r = np.hypot(x, y)
    # Relative velocity: components and magnitude
    wx = u - ugas
    wy = v - vgas
    w = np.hypot(wx, wy)

    # Magnitude of radiative acceleration
    a_rad = 0.5/r**2
    # Magnitude of drag 
    a_drag = 0.5*s.drag_constant*ds79.Fdrag(w*s.vinf, s.T, phi(r, s))
    # a_rad is directed along +r_hat ...
    ax = a_rad*x/r
    ay = a_rad*y/r
    if w > 0.0:
        # ... while a_drag is along -w_hat
        ax += -a_drag*wx/w
        ay += -a_drag*wy/w
    return ax, ay

def vector_accel_3d_frozen(x, y, z, vx, vy, vz, s,
                           vxgas=-1.0, vygas=0.0, vzgas=0.0):
    """
    Total 3d vector acceleration on a grain: radiative + drag,
    assuming that Lorentz force is so strong that grains can only be
    accelerated along the field lines, not perpendicular to them

    Input parameters: `x`, `y`, `z` are position in units of R_**
    (drag-free turnaround radius).  `vx`, `vy`, `vz` are velocity in
    units of v_inf.  `s` is a `DustStream` object, which must contain
    the unit vector `s.b_hat` along the B-field.  Optional arguments
    `vxgas`, `vygas`, `vzgas` are for if we do the back reaction on
    the gas.

    Returns: `ax`, `ay`, `az`, the cartesian components of
    acceleration
    """
    # Radius from star
    r = np.sqrt(x**2 + y**2 + z**2)
    # Unit vector in radial direction
    r_hat = np.array([x/r, y/r, z/r])
    # Magnitude of radiative acceleration
    a_rad = 0.5/r**2
    # Contribution of radiation to acceleration vector 
    a_vec = a_rad*r_hat

    # Relative velocity: components and magnitude
    wx = vx - vxgas
    wy = vy - vygas
    wz = vz - vzgas
    w = np.sqrt(wx**2 + wy**2 + wz**2)
    if w > 0.0:
        # Unit vector along slip velocity
        w_hat = np.array([wx/w, wy/w, wz/w])
        # Magnitude of drag 
        a_drag = 0.5*s.drag_constant*ds79.Fdrag(w*s.vinf, s.T, phi(r, s))
        # Contribution of drag to acceleration vector
        a_vec += -a_drag*w_hat

    # Only retain component projected along B-field
    a_proj = np.dot(a_vec, s.b_hat)
    ax, ay, az = a_proj*s.b_hat
    return ax, ay, az


def dydt_3d(state, t, s, accel_func=vector_accel_3d_frozen):
    """
    3D equation of motion for dust grain, subject to a variety of
    forces.  Used in ODE evolution of `state` vector with `t`

    Extra argument `s` is a DustStream() instance.  Optional argument
    is function that evaluates the vector sum of accelerations
    """
    x, vx, y, vy, z, vz = state
    dxdt = vx
    dydt = vy
    dzdt = vz
    dvxdt, dvydt, dvzdt = accel_func(x, y, z, vx, vy, vz, s)
    return [dxdt, dvxdt, dydt, dvydt, dzdt, dvzdt]



def phi(x, s):
    """
    Grain potential: phi = U / k T

    Input parameters: `x` is radius in units of R_** (drag-free
    turnaround radius).  `s` is a `DustStream` object.
    """
    # P_rad / P_gas
    Up = s.Upstarstar / x**2
    return 1.5*s.phi_norm*(np.log(Up) + 2.3)



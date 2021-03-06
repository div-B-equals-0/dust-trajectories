import sys
import json
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import seaborn as sns
import duststream as ds

PC = 3.085677582e18
KM = 1.0e5
YR = 3600*24*365.25
L4 = {
    "MS10": 0.64,
    "MS20": 5.45,
    "MS40": 22.2,
    "BSG": 30.2,
}

eta = {
    "MS10": 0.0066,
    "MS20": 0.1199,
    "MS40": 0.4468,
    "BSG": 0.3079,
}

# Set minor parameters according to star and grain type
Tgas = {
    "MS10": 8000.0,
    "MS20": 9000.0,
    "MS40": 1e4,
    "BSG": 8000.0,
}


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

        
class Trajectory3d(object):
    """
    Three-dimensional dust trajectory in the limit R_Larmor -> 0
    """
    def __init__(self, star, vinf, logn, grain, a, thetaB):
        """
        Initialize stream for given impact parameter, star, and dust
        parameters
        """
        self.star = star
        self.grain = grain
        self.id_ = f"{star}-v{int(vinf):03d}-n{int(10*logn):+03d}"
        self.id_ += f"-{grain}{int(100*a):03d}"
        self.id_ += f"-th{int(thetaB):03d}"

        # Solid body density depends on type of grain, while
        # normalization of potential phi depends on both grain type
        # and hardness of radiation field
        if grain == "gra":
            rho_d = 2.2
            if star in ["MS10", "BSG"]:
                phi_norm = 1.0
            else:
                phi_norm = 1.5
        else:
            rho_d = 3.5
            if star in ["MS10", "BSG"]:
                phi_norm = 0.7
            else:
                phi_norm = 1.4

        self.stream = ds.DustStream(L4=L4[star], vinf=vinf, n=10**logn, a=a,
                                    thetaB=thetaB,
                                    eta=eta[star], T=Tgas[star],
                                    phi_norm=phi_norm, rho_d=rho_d)
        self.stash_basic_data()
        # Pre-calculate the map of net acceleration vs (R, w)
        self.make_accel_map()
        # We need to find the sonic drift radius before doing integration
        self.analyze_accel_map()


    def integrate(self, Y0=0.0, Z0=0.0, Xstart=10.0, tstop=40.0, nt=5001):
        """
        Solve the ODE to find the grain trajectory `X`, `V` versus `t`

        These are in units of R_sd and v_inf
        """
        self.Y0 = Y0
        self.Z0 = Z0
        
        # Externally, we use R_sd and Rs_d / v_inf as characteristic
        # scales for distance and time, respectively.  Internally, the
        # integration uses R_**, so we must convert between them:
        rescale = self.Rsd / self.stream.Rstarstar
        
        # Initial conditions in units of R_** and v_inf
        y0 = [Xstart*rescale, -1.0,
              self.Y0*rescale, 0.0,
              self.Z0*rescale, 0.0]
        # Time grid
        self.Time = np.linspace(0.0, tstop*rescale, nt)
        soln = odeint(ds.dydt_3d, y0, self.Time, args=(self.stream,))
        # X, V, and W are dimensionless 
        self.X = soln[:, 0]
        self.VX = soln[:, 1]
        self.Y = soln[:, 2]
        self.VY = soln[:, 3]
        self.Z = soln[:, 4]
        self.VZ = soln[:, 5]
        self.W = np.sqrt((self.VX + 1.0)**2
                         + self.VY**2 + self.VZ**2)
        self.Vmag = np.sqrt(self.VX**2 + self.VY**2 + self.VZ**2)
        self.RR = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        # Radial velocity
        self.Vrad = (self.VX*self.X
                     + self.VY*self.Y
                     + self.VZ*self.Z)/self.RR

        # R, v, and w are dimensional
        self.x = self.X*self.stream.Rstarstar
        self.y = self.Y*self.stream.Rstarstar
        self.z = self.Z*self.stream.Rstarstar
        self.R = self.RR*self.stream.Rstarstar
        self.vx = self.VX*self.stream.vinf
        self.vy = self.VY*self.stream.vinf
        self.vz = self.VZ*self.stream.vinf
        # And put time in years
        self.tstarstar = (PC*self.stream.Rstarstar) / (KM*self.stream.vinf)
        self.tstarstar /= YR
        self.t = self.Time*self.tstarstar
        
        # Now analyze trajectory for diagnostic data
        self.analyze_trajectory()

    def stash_basic_data(self):
        """
        Populate dict of data about this run, for saving as json later
        """
        self.data = {
            "star": self.star,
            "grain": self.grain,
            "stream": self.stream.__dict__
        }
        
        
    def analyze_accel_map(self):
        """
        Find radii where drift velocity has interesting values
        """
        # Loop over drift velocity grid, finding smallest radius where
        # net acceleration is zero
        R_wd_pts = np.empty_like(self.wpts)
        for j in range(len(R_wd_pts)):
            try:
                R_wd_pts[j] = self.Rpts[self.amap[j, :] < 0.0].min()
            except ValueError:
                # Can happen for low density and low w
                R_wd_pts[j] = 2*self.Rpts[-1]
        # Sonic drift radius is minimum of these radii for w not too large
        idx = np.where(self.wpts < 100.0, R_wd_pts, 999.0).argmin()
        self.Rsd = R_wd_pts[idx]
        self.data["R sonic drift"] = self.Rsd
        self.data["w sonic drift"] = self.wpts[idx]
        # Critical radius (stable/unstable boundary) is max of radii
        # for w not too small
        idx = np.where(self.wpts > 20.0, R_wd_pts, 0.0).argmax()
        self.data["R critical"] = R_wd_pts[idx]
        self.data["w critical"] = self.wpts[idx]
        # Equilibrium radius where w_drift = v_inf
        self.Req = np.interp(self.stream.vinf, self.wpts, R_wd_pts)
        self.data["R equilib drift"] =  self.Req
        self.data["w equilib drift"] = self.stream.vinf
        # self.data["w drift grid"] = self.wpts[::5].tolist()
        # self.data["R drift grid"] = R_wd_pts[::5].tolist()
        
    def analyze_trajectory(self):
        self.data["Y0"] = self.Y0
        self.data["Z0"] = self.Z0

        # Mask of those points where radial component of velocity
        # changes sign between this point and the next
        crossings = self.Vrad[:-1]*self.Vrad[1:] < 0
        # Extend to size of original arrays
        crossings = np.concatenate((crossings, [False]))
        # Mask of R maxima, where V_rad goes from + to - 
        maxima = crossings & (self.Vrad > 0.0)
        # Mask of R minima, where V_rad goes from - to + 
        minima = crossings & (self.Vrad < 0.0)

        # Save lists of all t and R for each type of extrema 
        self.data["t minima"] = self.t[minima] 
        self.data["R minima"] = self.R[minima] 
        self.data["X minima"] = self.X[minima] 
        self.data["Y minima"] = self.Y[minima] 
        self.data["Z minima"] = self.Z[minima] 

        self.data["t maxima"] = self.t[maxima] 
        self.data["R maxima"] = self.R[maxima] 
        self.data["X maxima"] = self.X[maxima] 
        self.data["Y maxima"] = self.Y[maxima] 
        self.data["Z maxima"] = self.Z[maxima] 

        # Final state
        self.data["X end"] = self.X[-1]
        self.data["Y end"] = self.Y[-1]
        self.data["Z end"] = self.Z[-1]
        self.data["t end"] = self.t[-1]
        
        # Biggest positive velocity
        self.data["v max"] = self.vx.max()

    def stash_trajectory(self):
        self.data["trajectory"] = {
            k: getattr(self, k)
            for k in ["Time", "X", "Y", "Z", "VX", "VY", "VZ"] 
        }
        
    def savedata(self, prefix):
        self.datafile = f'{prefix}-{self.id_}.json'
        with open(self.datafile, "w") as f:
            json.dump(self.data, f, indent=4, cls=MyEncoder)

    def make_accel_map(self):
        """
        Map of net acceleration (radiative plus drag) on a grid in
        physical units of radius (pc) vs gas-grain velocity difference
        (km/s)
        """
        self.Rlim = 2e-4, 20.0
        self.wlim = 0.03, 2000
        self.Rpts = np.logspace(
            np.log10(self.Rlim[0]), np.log10(self.Rlim[1]), 201)
        self.wpts = np.logspace(
            np.log10(self.wlim[0]), np.log10(self.wlim[1]), 201)
        # total_accel expects radius in units of R**, but velocity in km/s
        self.amap = ds.total_accel(
            self.Rpts[None, :]/self.stream.Rstarstar, self.wpts[:, None],
            self.stream)
        
    def savefig(self, prefix):
        """
        Make a nice phase-space plot of the trajectory
        """
        self.figfile = f'{prefix}-{self.id_}.pdf'

        t0 = self.t[self.Vrad >= 0.0].min()

        sns.set_style('ticks')
        sns.set_color_codes('deep')
        fig, (ax, axp) = plt.subplots(2, 1, figsize=(4, 6))
        ax.plot(self.t - t0, self.x/self.Rsd, label=r'$x/R_\dag$')
        ax.plot(self.t - t0, self.VX, label='$v_x / v_{\infty}$', lw=1.0)
        ax.plot(self.t - t0, self.y/self.Rsd, label=r'$y/R_\dag$')
        ax.plot(self.t - t0, self.VY, label='$v_y / v_{\infty}$', lw=1.0)
        #ax.plot(t - t0, wdrift, ls='--', label='$w_\mathrm{drift} / v_{\infty}$')

        ax.axhspan(0.0, 1.0, color='k', alpha=0.1)
        ax.legend(loc="upper right")
        ax.set(
            xlabel=r'Time, years',
            ylim=[-1.5, 3.1],
        )

        # Acceleration map was already created during initialization 
        # axp.contour(self.Rpts, self.wpts, self.amap, [0.0],
        #             linewidths=3, linestyles=":", colors="m")
        # for z, cmap, dex in [[np.log10(self.amap), "Blues", 10.0],
        #                      [np.log10(-self.amap), "Reds", 4.0]]: 
        #     axp.contourf(self.Rpts, self.wpts, z, 10,
        #                  vmax=np.nanmax(z), vmin=np.nanmax(z)-dex, cmap=cmap)

        axp.plot(self.x, self.vx, lw=1, label='$(x, v_x)$')
        axp.plot(self.y, self.vy, lw=1, label='$(y, v_y)$')
        axp.axhline(self.stream.vinf, color='k', lw=0.5)
        axp.axhline(0.0, color='k', lw=0.5)
        axp.axhline(-self.stream.vinf, color='k', lw=0.5)
        axp.axvline(0.0, color='k', lw=0.5)
        axp.axvline(self.Rsd, color='k', lw=0.5)
        axp.axvline(self.stream.Rstarstar, color='k', ls=":", lw=0.5)
        axp.axvline(self.stream.R0, color='r', lw=1, ls="--")
        axp.set(xlabel='$x, y$, pc', ylabel='$v_x, v_y$, km/s',
                xlim=[-3*self.Rsd, 3*self.Rsd],
                ylim=[-1.3*self.stream.vinf, 3.1*self.stream.vinf],
                xscale="linear", yscale="linear",
                xticks=0.5*np.arange(7),
                yticks=[-1.0, -0.5, 0., 0.5, 1.0, 1.5])

        sns.despine()
        fig.tight_layout()
        fig.savefig(self.figfile)
        # Prevent resource leaks
        plt.close(fig)
        

if __name__ == "__main__":

    try:
        STAR = sys.argv[1]
        VINF = float(sys.argv[2])
        LOGN = float(sys.argv[3])
        GRAIN = sys.argv[4]
        A = float(sys.argv[5])
        THETAB = float(sys.argv[6])
        Y0 = float(sys.argv[7])
        Z0 = float(sys.argv[8])
    except:
        print(f"Usage: {sys.argv[0]} STAR VINF LOGN GRAIN A THETAB Y0 Z0")


    traj = Trajectory3d(STAR, VINF, LOGN, GRAIN, A, THETAB)
    traj.integrate(Y0, Z0, Xstart=3.0, tstop=20.0)
    yz_id = f"Y{int(1000*Y0):04d}-Z{int(1000*Z0):04d}"
    traj.savedata(f"data/frozen-{yz_id}")
    traj.savefig(f"figs/frozen-{yz_id}")
    print(traj.figfile, end='')






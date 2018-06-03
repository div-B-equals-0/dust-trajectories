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

        
class Trajectory(object):
    """
    Dust trajectory on axis
    """
    def __init__(self, star, vinf, logn, grain, a):
        """
        Initialize stream for given star and dust parameters
        """
        self.star = star
        self.grain = grain
        self.id_ = f"{star}-v{int(vinf):03d}-n{int(10*logn):+03d}"
        self.id_ += f"-{grain}{int(100*a):03d}"

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
                                    eta=eta[star], T=Tgas[star],
                                    phi_norm=phi_norm, rho_d=rho_d)
        self.stash_basic_data()
        # Pre-calculate the map of net acceleration vs (R, w)
        self.make_accel_map()
        # We need to find the sonic drift radius before doing integration
        self.analyze_accel_map()


    def integrate(self, Rstart=10.0, tstop=40.0, nt=5001):
        """
        Solve the ODE to find the grain trajectory `X`, `V` versus `t`

        These are in units of R** and v_inf
        """

        # Use Rsd as a characteristic scale for distance and time
        rescale = self.Rsd / self.stream.Rstarstar
        
        # Initial conditions
        y0 = [Rstart*rescale, -1.0]
        # Time grid
        self.Time = np.linspace(0.0, tstop*rescale, nt)
        soln = odeint(ds.dydt_1d, y0, self.Time, args=(self.stream,))
        # X, V, and W are dimensionless 
        self.X = soln[:, 0]
        self.V = soln[:, 1]
        self.W = 1.0 + self.V

        # R, v, and w are dimensional
        self.R = self.X*self.stream.Rstarstar
        self.v = self.V*self.stream.vinf
        self.w = self.W*self.stream.vinf
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
        # self.data["w drift grid"] = self.wpts[::5]
        # self.data["R drift grid"] = R_wd_pts[::5]
        
    def analyze_trajectory(self):
        # Mask of those points where V changes sign between this point
        # and the next
        crossings = self.V[:-1]*self.V[1:] < 0
        # Extend to size of original arrays
        crossings = np.concatenate((crossings, [False]))
        # Mask of R maxima, where V goes from + to - 
        maxima = crossings & (self.V > 0.0)
        # Mask of R minima, where V goes from - to + 
        minima = crossings & (self.V < 0.0)

        # Save lists of all t and R for each type of extrema 
        self.data["t min"] = self.t[minima] 
        self.data["R min"] = self.R[minima] 
        self.data["t max"] = self.t[maxima] 
        self.data["R max"] = self.R[maxima] 

        # Final state
        self.data["R end"] = self.R[-1]
        self.data["v end"] = self.v[-1]
        self.data["t end"] = self.t[-1]
        
        # Biggest positive velocity
        self.data["v max"] = self.v.max()
        
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

        t0 = self.t[self.V >= 0.0].min()

        sns.set_style('ticks')
        sns.set_color_codes('deep')
        fig, (ax, axp) = plt.subplots(2, 1, figsize=(4, 6))
        ax.plot(self.t - t0, self.R/self.Rsd, label=r'$R/R_\mathrm{sd}$')
        ax.plot(self.t - t0, self.V, label='$v / v_{\infty}$', lw=1.0)
        #ax.plot(t - t0, wdrift, ls='--', label='$w_\mathrm{drift} / v_{\infty}$')

        ax.axhspan(0.0, 1.0, color='k', alpha=0.1)
        ax.legend(loc="upper right")
        ax.set(
            xlabel=r'Time, years',
            ylim=[-1.5, 5.1]
        )

        # Acceleration map was already created during initialization 
        axp.contour(self.Rpts, self.wpts, self.amap, [0.0],
                    linewidths=3, linestyles=":", colors="m")
        for z, cmap, dex in [[np.log10(self.amap), "Blues", 10.0],
                             [np.log10(-self.amap), "Reds", 4.0]]: 
            axp.contourf(self.Rpts, self.wpts, z, 10,
                         vmax=np.nanmax(z), vmin=np.nanmax(z)-dex, cmap=cmap)

        axp.plot(self.R, self.w, lw=4, color="w", alpha=0.5)
        axp.plot(self.R, self.w, lw=2, color="k", alpha=1.0)
        axp.axhline(self.stream.vinf, color='k', lw=0.5)
        axp.axvline(self.stream.Rstarstar, color='k', lw=0.5)
        axp.axvline(self.stream.R0, color='r', lw=2, ls="--")
        axp.set(xlabel='$R$, pc', ylabel='$w$, km/s',
                xlim=self.Rlim, ylim=self.wlim,
                xscale="log", yscale="log",
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
    except:
        print(f"Usage: {sys.argv[0]} L4 VINF LOGN GRAIN A")


    traj = Trajectory(STAR, VINF, LOGN, GRAIN, A)
    traj.integrate()
    traj.savedata("data/on-axis")
    traj.savefig("figs/on-axis")
    print(traj.figfile, end='')






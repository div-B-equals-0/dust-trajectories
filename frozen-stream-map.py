import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from frozen_trajectory import Trajectory3d

try:
    STAR = sys.argv[1]
    VINF = float(sys.argv[2])
    LOGN = float(sys.argv[3])
    GRAIN = sys.argv[4]
    A = float(sys.argv[5])
    THETAB = float(sys.argv[6])
    Z0 = float(sys.argv[7])
except:
    sys.exit(f"Usage: {sys.argv[0]} STAR VINF LOGN GRAIN A THETAB Z0 [ZOOM]")

# Inertia-confined case is small when using R_\dag (Rsd) as the
# distance scale, so allow for optional zooming in
try:
    ZOOM = float(sys.argv[8])
except:
    ZOOM = 1.0

stream_id = f"{STAR}-v{int(VINF):03d}-n{int(10*LOGN):+03d}"
stream_id += f"-{GRAIN}{int(100*A):03d}"
stream_id += f"-th{int(THETAB):03d}"
stream_id += f"-Z{int(1000*Z0):04d}"

traj = Trajectory3d(STAR, VINF, LOGN, GRAIN, A, THETAB)
xmin, xmax = -3*traj.Rsd/ZOOM, 3*traj.Rsd/ZOOM
ymin, ymax = -3*traj.Rsd/ZOOM, 3*traj.Rsd/ZOOM

# Impact parameter grid
N = 1
ny0 = N*400 + 1
y0grid = 0.001 + np.linspace(ymin/traj.Rsd, ymax/traj.Rsd, ny0)

# Grid to save R(theta)
nth = 200
thm_grid = np.linspace(-np.pi, np.pi, nth)

# Wilkinoid for showing bow shock
rm = np.sqrt(3*(1 - thm_grid/np.tan(thm_grid)))/np.abs(np.sin(thm_grid))

# (x, y) coordinates are in parsec
# w is weight because of geometry
# wp is extra weight for plotting
xx, yy, ww, wp = [], [], [], []
xs, ys = [], []
nt, tstop, Xstart = 1001, 30.0, 3.0/ZOOM
for iy0, y0 in enumerate(y0grid[::-1]):
    traj.integrate(y0, Z0, Xstart=Xstart, tstop=tstop, nt=nt)

    tfine = np.linspace(traj.t[0], traj.t[-1], 10*nt)
    xfine = np.interp(tfine, traj.t, traj.x)
    yfine = np.interp(tfine, traj.t, traj.y)
    # Accumulate (x, y) points in a long list
    xx.extend(xfine)
    yy.extend(yfine)
    # Now slab symmetry, so natural weights are unity
    weight = np.ones_like(xfine)
    ww.extend(weight)
    # De-accentuate large radii for plotting
    r = np.hypot(xfine, yfine)
    wp.extend(weight/r)
    
    if iy0 % 10*N == 5*N:
        # Save streamlines for selected impact parameters
        xs.append(traj.x)
        ys.append(traj.y)

# Use the last trajectory for getting the model parameters


figfile = sys.argv[0].replace(".py", "-" + stream_id + ".pdf")
sns.set_color_codes()
fig, ax = plt.subplots(figsize=(5, 5))

# Plot a density histogram of all the (x, y) points we accumulated
H, xe, ye = np.histogram2d(xx, yy, bins=(200, 200), weights=wp,
                           range=[[xmin, xmax], [ymin, ymax]])
Hd, xe, ye = np.histogram2d(xx, yy, bins=(200, 200), weights=ww,
                           range=[[xmin, xmax], [ymin, ymax]])
rho_med = np.median(Hd)
rho_max = Hd.max()
# rho_scale = np.sqrt(rho_med*rho_max)
rho_scale = 10*rho_med
ax.imshow(Hd.T, origin='lower', extent=[xmin, xmax, ymin, ymax],
          vmin=0.0, vmax=rho_scale, cmap='gray_r')

# Scatter-shot grain cohorts to show evolution
x1s = [2.8, 1.4, 0, -1.4, -2.8]
# Bespoke collection of colors for the scatter shot grains at each x1 position
colors = [
    # First batch at x=4 is separated from rest
    # Pale yellow
    (0.8, 0.8, 0.3, 1.0),
    # Swing towards orange for next two at x=2 and x=0
    (0.9, 0.7, 0.2, 1.0),
    (1.0, 0.4, 0.1, 1.0),
    # Now we need a contrast - go more purple
    (0.7, 0.1, 0.4, 1.0),
    # And finally, more blue
    (0.2, 0.2, 0.5, 1.0)
]


# Plot the streamlines that we saved earlier
for x, y in zip(xs, ys):
    ax.plot(x, y, '-', color='w', lw=1.4, alpha=0.7)
    ax.plot(x, y, '-', color='k', lw=0.8)
    for x1, color in zip(x1s, colors):
        itime = int((Xstart - x1/ZOOM)*nt/tstop)
        dtime = int(5/ZOOM)
        ax.plot(x[itime-dtime:itime+2*dtime:dtime],
                y[itime-dtime:itime+2*dtime:dtime],
                '.', ms=4.0, color=color, zorder=20-x1)


# Plot the B-field lines
cthB = np.cos(np.deg2rad(THETAB))
sthB = np.sin(np.deg2rad(THETAB))
for xx in np.linspace(1.5*xmin, 1.5*xmax, 15):
    yy1, yy2 = 1.5*ymin, 1.5*ymax
    x1 = -xx*sthB + yy1*cthB
    x2 = -xx*sthB + yy2*cthB
    y1 = xx*cthB + yy1*sthB
    y2 = xx*cthB + yy2*sthB
    ax.plot([x1, x2], [y1, y2], lw=2, alpha=0.5, color='c')

    
    
# Plot inner bow shock as a Wilkinoid
ax.plot(traj.stream.R0*rm*np.cos(thm_grid),
        traj.stream.R0*rm*np.sin(thm_grid),
        ':', color='k', alpha=0.5, lw=2)

# Plot R_dag: rip radius
ax.plot(traj.Rsd*np.cos(thm_grid),
        traj.Rsd*np.sin(thm_grid),
        '--', color='r', alpha=1.0, lw=2)

# Plot equilibrium drift radius
ax.plot(traj.data["R equilib drift"]*np.cos(thm_grid),
        traj.data["R equilib drift"]*np.sin(thm_grid),
        '--', color='c', alpha=1.0, lw=2)

# Plot drag-free turnaround radius
ax.plot(traj.stream.Rstarstar*np.cos(thm_grid),
        traj.stream.Rstarstar*np.sin(thm_grid),
        '--', color='g', alpha=1.0, lw=2)

# Mark the axes and origin
ax.axvline(0.0, ls='--', color='0.5', lw=0.5)
ax.axhline(0.0, ls='--', color='0.5', lw=0.5)
ax.plot([0.0], [0.0], '+', color='k')
ax.plot([0.0], [0.0], '.', color='k')

ax.set_aspect('equal', adjustable='box-forced')

ax.set(
    xlim=[xmin, xmax],
    ylim=[ymin, ymax],
    xlabel="$x$, pc", 
    ylabel="$y$, pc", 
)

sns.despine()
fig.tight_layout()

fig.savefig(figfile)
print(figfile, end='')


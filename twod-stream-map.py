import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from twod_traject import Trajectory2d

try:
    STAR = sys.argv[1]
    VINF = float(sys.argv[2])
    LOGN = float(sys.argv[3])
    GRAIN = sys.argv[4]
    A = float(sys.argv[5])
except:
    print(f"Usage: {sys.argv[0]} STAR VINF LOGN GRAIN A")

stream_id = f"{STAR}-v{int(VINF):03d}-n{int(10*LOGN):+03d}"
stream_id += f"-{GRAIN}{int(100*A):03d}"

# Impact parameter grid
N = 5
ny0 = N*200 + 1
y0grid = 0.001 + np.linspace(0.0, 4.0, ny0)

# Grid to save R(theta)
nth = 200
thm_grid = np.linspace(0.0, np.pi, nth)
dth = np.pi/nth

# Parabola for showing bow shock
rm = 2.0/(1.0 + np.cos(thm_grid))
xlocus = rm*np.cos(thm_grid)
ylocus = rm*np.sin(thm_grid)

# (x, y) coordinates are in parsec
# w is weight because of geometry
# wp is extra weight for plotting
xx, yy, ww, wp = [], [], [], []
xs, ys = [], []
for iy0, y0 in enumerate(y0grid[::-1]):
    traj = Trajectory2d(y0, STAR, VINF, LOGN, GRAIN, A)
    traj.integrate(Xstart=5.0, tstop=20.0)

    # Accumulate (x, y) points in a long list
    xx.extend(traj.x)
    yy.extend(traj.y)
    # In cylindrical symmetry, weights proportional to 1/y
    weight = traj.y[0]/traj.y
    ww.extend(weight)
    # De-accentuate large radii for plotting
    r = np.hypot(traj.x, traj.y)
    wp.extend(weight/r)
    
    if iy0 % 30 == 15:
        # Save streamlines for selected impact parameters
        xs.append(traj.x)
        ys.append(traj.y)

# Use the last trajectory for getting the model parameters

xmin, xmax = -2.99*traj.Rsd, 2.99*traj.Rsd
ymin, ymax = 0.0, 3.99*traj.Rsd

figfile = sys.argv[0].replace(".py", "-" + stream_id + ".pdf")
sns.set_color_codes()
fig, ax = plt.subplots()

# Plot a density histogram of all the (x, y) points we accumulated
H, xe, ye = np.histogram2d(xx, yy, bins=(80/1, 50/1), weights=wp,
                           range=[[xmin, xmax], [ymin, ymax]])
Hd, xe, ye = np.histogram2d(xx, yy, bins=(80/1, 50/1), weights=ww,
                           range=[[xmin, xmax], [ymin, ymax]])
rho_med = np.median(H)
rho_max = H.max()
# rho_scale = np.sqrt(rho_med*rho_max)
rho_scale = rho_max
ax.imshow(H.T, origin='lower', extent=[xmin, xmax, ymin, ymax],
          vmin=0.0, vmax=rho_scale, cmap='gray_r')
# Plot the streamlines that we saved earlier
for x, y in zip(xs, ys):
    ax.plot(x, y, '-', color='w', lw=2.2, alpha=0.7)
    ax.plot(x, y, '-', color='k', lw=1.0)

# Plot inner bow shock as a parabola, but with Pi = Lambda = 1.7
ax.plot(traj.stream.R0*rm*np.cos(thm_grid),
        (1.7/2.0)*traj.stream.R0*rm*np.sin(thm_grid),
        ':', color='k', alpha=0.5, lw=2)

# Plot R_dag: rip radius
ax.plot(traj.Rsd*np.cos(thm_grid),
        traj.Rsd*np.sin(thm_grid),
        '--', color='r', alpha=1.0, lw=2)

# Plot equilibrium drift radius
ax.plot(traj.data["R equilib drift"]*np.cos(thm_grid),
        traj.data["R equilib drift"]*np.sin(thm_grid),
        '--', color='c', alpha=1.0, lw=2)


ax.axvline(0.0, ls='--', color='k', lw=0.5)
ax.set_aspect('equal', adjustable='box-forced')

ax.set(
    xlim=[xmin, xmax],
    ylim=[ymin, ymax],
    xlabel="$x$, pc", 
    ylabel="$y$, pc", 
)

fig.savefig(figfile)
print(figfile, end='')


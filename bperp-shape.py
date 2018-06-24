import sys
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import seaborn as sns


def rhs(y, t):
    """Right-hand side of ODE"""
    ydot = y[1]
    ydotdot = 0.5 * y[0] / (y[0]**2 + t**2)**1.5
    return ydot, ydotdot

def trajectory(tgrid, y0, ydot0=0):
    """Find dust grain trajectory tied to B field - no drag"""
    soln = odeint(rhs, [y0, ydot0], tgrid)
    return soln[:, 0], soln[:, 1]


figfile = sys.argv[0].replace(".py", ".png")

y0max = 10.0
ny0 = 4000
y0min = y0max/ny0
y0grid_lin = np.linspace(y0min, y0max, ny0)
ny0_log = 50
y0grid_log = np.logspace(np.log10(y0min) - 4.0, np.log10(y0min), ny0_log)
y0grid = np.concatenate([y0grid_log, y0grid_lin])
jsamples = [ny0_log + int(ny0*y/y0max) for y in [y0min, 0.2, 1.0, 2.0]]



nt = 32001

sns.set_color_codes()
fig, ax = plt.subplots(figsize=(5, 4))
ystack, vstack, xstack, wstack = [], [], [], []
tgrid = np.linspace(-20, 10, nt)
dt = tgrid[1] - tgrid[0]
for y0 in y0grid:
    t = tgrid + np.random.random(len(tgrid))*0.8*dt
    y, v = trajectory(t, y0)
    ystack.append(y)
    vstack.append(v)
    xstack.append(-t)
    if y0 in y0grid_lin:
        # weight by 1/r
        wstack.append(y0/y)
    else:
        # do not include streamlines from log grid
        wstack.append(np.zeros_like(y))

# Find inner envelope of trajectories: minimum y at each x
ystack = np.array(ystack)
vstack = np.array(vstack)
xstack = np.array(xstack)
wstack = np.array(wstack)
yshape = np.min(ystack, axis=0)

# Fit a second order polynomial to x(y)
m = yshape > 0.5
p2 = np.poly1d(np.polyfit(yshape[m], tgrid[m], 2))

# Fit a first order polynomial to x(y**2)
p1 = np.poly1d(np.polyfit(yshape[m]**2, tgrid[m], 1))

# extended y grid to show negative side
yext = np.linspace(-2.0, yshape.max(), 200)
j0 = np.argmin(p2(yext))

# find density by binning
xmin, xmax = -8.0, 1.5
ymin, ymax = -1.2, 7.0
H, xe, ye = np.histogram2d(xstack.ravel(), ystack.ravel(),
                           bins=(4*95, 4*82), weights=wstack.ravel(),
                           range=[[xmin, xmax], [ymin, ymax]])
Hm, xe, ye = np.histogram2d(xstack.ravel(), -ystack.ravel(),
                            bins=(4*95, 4*82), weights=wstack.ravel(),
                            range=[[xmin, xmax], [ymin, ymax]])

H += Hm
H0 = H[-1, -1]
ax.imshow(H.T, origin='lower', extent=[xmin, xmax, ymin, ymax],
          vmin=0.0, vmax=3*H0, cmap='gray_r')


ax.plot(-tgrid, yshape, lw=0.0)

ax.plot(-p1(yext**2), yext, lw=0.5, ls="--")
ax.plot(-p2(yext), yext, lw=0.5, ls="--")

ax.axhline(0.0, color="k", alpha=0.5, lw=0.3)

ax.plot(-p2(yext[j0]), yext[j0], "+", color="g", ms=4)
ax.plot(-p1(0.0), 0.0, "+", color="orange", ms=4)
ax.plot(0.0, 0.0, "*", color="r", ms=6)

for j in jsamples:
    ax.plot(-tgrid, ystack[j, :], lw=0.3, color="k")

#ax.legend(fontsize="small", ncol=2)

ax.set_aspect("equal")
ax.set(xlim=[-8.0, 1.5], ylim=[-1.2, 7.0],
       xlabel="$x / R_{**}$", ylabel="$y / R_{**}$")
sns.despine()
fig.tight_layout()
fig.savefig(figfile, dpi=600)

print(figfile, end="")

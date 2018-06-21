import sys
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import seaborn as sns


def rhs(y, t):
    """Right-hand side of ODE"""
    ydot = y[1]
    ydotdot = y[0] / (y[0]**2 + t**2)**1.5
    return ydot, ydotdot

def trajectory(tgrid, y0, ydot0=0):
    """Find dust grain trajectory tied to B field - no drag"""
    soln = odeint(rhs, [y0, ydot0], tgrid)
    return soln[:, 0], soln[:, 1]


figfile = sys.argv[0].replace(".py", ".pdf")

y0max = 20.0
ny0 = 1000
y0min = y0max/ny0
y0grid_lin = np.linspace(y0min, y0max, ny0)
ny0_log = 50
y0grid_log = np.logspace(np.log10(y0min) - 4.0, np.log10(y0min), ny0_log)
y0grid = np.concatenate([y0grid_log, y0grid_lin])
jsamples = [ny0_log + int(ny0*y/y0max) for y in [y0min, 0.2, 1.0, 2.0]]



nt = 8001

sns.set_color_codes()
fig, ax = plt.subplots(figsize=(5, 4))
ystack, vstack = [], []
tgrid = np.linspace(-20, 10, nt)
for y0 in y0grid:
    y, v = trajectory(tgrid, y0)
    ystack.append(y)
    vstack.append(v)

# Find inner envelope of trajectories: minimum y at each x
ystack = np.array(ystack)
vstack = np.array(vstack)
yshape = np.min(ystack, axis=0)

# Fit a second order polynomial to x(y)
m = yshape > 0.5
p2 = np.poly1d(np.polyfit(yshape[m], tgrid[m], 2))

# Fit a second order polynomial to x(y**2)
p1 = np.poly1d(np.polyfit(yshape[m]**2, tgrid[m], 1))

# extended y grid to show negative side
yext = np.linspace(-2.0, yshape.max(), 200)
j0 = np.argmin(p2(yext))

ax.plot(-tgrid, yshape)
#ax.plot(tgrid, np.sqrt(6*tgrid), lw=0.5)
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
ax.set(xlim=[-8.0, 1.5], ylim=[-1.2, 7.0], xlabel="$x$", ylabel="$y$")
sns.despine()
fig.tight_layout()
fig.savefig(figfile)

print(figfile, end="")

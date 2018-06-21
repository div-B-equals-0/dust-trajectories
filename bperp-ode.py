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

def tfit(y0):
    return 2000*np.log10(10000/y0)**-5
def afit(y0):
    return 1e-5*np.log10(3000/y0)**8.5

figfile = sys.argv[0].replace(".py", ".pdf")

y0min = 1e-6
y0max = 100.0
ny0 = 301
y0grid = np.logspace(np.log10(y0min), np.log10(y0max), ny0)

nt = 200001

fig, ax = plt.subplots(figsize=(5, 4))
vmax, tmax, twidth, yamax, amax = [], [], [], [], []
for y0 in y0grid:
    tscale = 30*tfit(y0)
    tgrid = np.linspace(-200, 200, nt)
    tgrid = np.linspace(-tscale, tscale, nt)
    y, v = trajectory(tgrid, y0)
    a = np.gradient(v, tgrid)
    i0 = a.argmax()
    tmax.append(-tgrid[i0])
    vmax.append(v[-1])
    yamax.append(y[i0] - y0)
    amax.append(a.max())
    t1 = tgrid[np.argmin(np.abs(a[:i0] - 0.5*a.max()))]
    t2 = tgrid[i0:][np.argmin(np.abs(a[i0:] - 0.5*a.max()))]
    twidth.append(t2 - t1)


x = y0grid
ax.plot(x, amax, label=r"$a_{\mathrm{max}}$")
ax.plot(x, vmax, label=r"$v_{\mathrm{final}}$")
ax.plot(x, yamax, label=r"$y(a_{\mathrm{max}}) - y_{0}$")
ax.plot(x, tmax, label=r"$-t(a_{\mathrm{max}})$")
ax.plot(x, twidth, label=r"$\Delta t$")
ax.plot(x, np.array(amax)*np.array(twidth),
        label=r"$a_{\mathrm{max}} \times \Delta t$")
# ax.plot(x, tfit(y0grid), lw=0.2, label="_nolabel_")
# ax.plot(x, afit(y0grid), lw=0.2, label="_nolabel_")
ax.axhline(1.0, alpha=0.4, color="k", lw=0.3)
ax.axvline(1.0, alpha=0.4, color="k", lw=0.3)
ax.legend(fontsize="small", ncol=2)
ax.set(xscale="log", yscale="log", xlabel="impact parameter, $y_{0}$")

sns.despine()
fig.tight_layout()
fig.savefig(figfile)

print(figfile, end="")

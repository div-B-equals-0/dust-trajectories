import sys
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import seaborn as sns
import duststream as ds

figfile = sys.argv[0].replace('.py', '.pdf')

# Initial conditions
y0 = [2.5, -1.0]

stream = ds.DustStream(vinf=20, phi=15.0, a=0.04, L4=0.63, n=1.0e-1)

# Time grid
t = np.linspace(0.0, 10.0, 501)
soln = odeint(ds.dydt_1d, y0, t, args=(stream,))
t0 = t[np.argmin(soln[:, 0])]

# Slippage velocity
w = 1.0 + soln[:, 1]
# Drift velocity
# wdrift = 1.0 / alpha / soln[:, 0]

sns.set_style('ticks')
sns.set_color_codes('dark')
fig, (ax, axp) = plt.subplots(2, 1, figsize=(4, 6))
ax.plot(t - t0, soln[:, 0], label='$R/R_{0}$')
ax.plot(t - t0, soln[:, 1], label='$v / v_{\infty}$')
#ax.plot(t - t0, wdrift, ls='--', label='$w_\mathrm{drift} / v_{\infty}$')

# ax.axhline(1.0/alpha, ls=':', color='k', lw=0.8)
ax.axhspan(0.0, 1.0, color='k', alpha=0.1)
ax.legend()
ax.set(
    xlabel=r'Time / $(R_{0} / v_{\infty})$',
    ylim=[-1.1, 2.1]
)
t2 = np.linspace(0.0, 20.0, 201)

x1, x2 = 0.9*soln[:, 0].min(), 3.0
w1, w2 = 1e-2, 3.0*w.max()

xpts = np.logspace(np.log10(x1), np.log10(x2), 151)
wpts = np.logspace(np.log10(w1), np.log10(w2), 101)

agrid = ds.total_accel(xpts[None, :], stream.vinf*wpts[:, None], stream)


# Add dimensions back in for plotting
xpts *= stream.Rstarstar
x1 *= stream.Rstarstar
x2 *= stream.Rstarstar

wpts *= stream.vinf
w1 *= stream.vinf
w2 *= stream.vinf

axp.contour(xpts, wpts, agrid, [0.0])
axp.contourf(xpts, wpts, np.log10(agrid),
            [-0.5, 0.0, 0.5, 1.0, 1.5], cmap="Reds")
axp.contourf(xpts, wpts, np.log10(-agrid),
            [-0.5, 0.0, 0.5, 1.0, 1.5], cmap="Blues")

axp.plot(soln[:, 0]*stream.Rstarstar, w*stream.vinf, lw=1.4, color="0.9")
axp.plot(soln[:, 0]*stream.Rstarstar, w*stream.vinf, lw=0.5, color="g")
axp.axhline(stream.vinf, color='k', lw=0.5)
axp.axvline(stream.Rstarstar, color='k', lw=0.5)
axp.set(xlabel='$R$, pc', ylabel='$w$, km/s',
        xlim=[x1, x2], ylim=[w1, w2],
        xscale="log", yscale="log",
        xticks=0.5*np.arange(7),
        yticks=[-1.0, -0.5, 0., 0.5, 1.0, 1.5])

sns.despine()
fig.tight_layout()
fig.text(0.02, 0.97, '(a)')
fig.text(0.02, 0.5, '(b)')
fig.savefig(figfile)
print(figfile, end='')

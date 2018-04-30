import sys
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import seaborn as sns
import duststream as ds

figfile = sys.argv[0].replace('.py', '.pdf')

# Initial conditions
y0 = [2.5, -1.0]

stream = ds.DustStream(vinf=70.0, phi=10.0, a=0.02, L4=10.0)

# Time grid
t = np.linspace(0.0, 4.0, 501)
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
ax.plot(t - t0, w, label='$w / v_{\infty}$')
#ax.plot(t - t0, wdrift, ls='--', label='$w_\mathrm{drift} / v_{\infty}$')

# ax.axhline(1.0/alpha, ls=':', color='k', lw=0.8)
ax.axhspan(0.0, 1.0, color='k', alpha=0.1)
ax.legend()
ax.set(
    xlabel=r'Time / $(R_{0} / v_{\infty})$',
    yscale="log",
    ylim=[0.001, 10.0]
)
t2 = np.linspace(0.0, 20.0, 201)

axp.plot(soln[:, 0], w, lw=0.1)
axp.axhline(1.0, xmax=0.75, color='k', lw=0.5)
axp.legend(title='Phase space\n  trajectories')
axp.set(xlabel='$R/R_{0}$', ylabel='$w / v_{\infty}$',
        xlim=[0.009, 2.5], ylim=[0.01, 10.0],
        xscale="log", yscale="log",
        xticks=0.5*np.arange(7),
        yticks=[-1.0, -0.5, 0., 0.5, 1.0, 1.5])

sns.despine(trim=True)
fig.tight_layout()
fig.text(0.02, 0.97, '(a)')
fig.text(0.02, 0.5, '(b)')
fig.savefig(figfile)
print(figfile, end='')

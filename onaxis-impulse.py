"""
Calculate total drag impulse on grain versus time, so that we can see
what the back reaction no the gas will be
"""
import sys
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from onaxis_traject import Trajectory
import duststream as ds
import ds79


try: 
    STAR = sys.argv[1]
    VINF = float(sys.argv[2])
    LOGN = float(sys.argv[3])
    GRAIN = sys.argv[4]
    A = float(sys.argv[5])
except:
    print(f"Usage: {sys.argv[0]} L4 VINF LOGN GRAIN A")


traj = Trajectory(STAR, VINF, LOGN, GRAIN, A)
traj.integrate(tstop=25, nt=10001)

# Now find drag force as function of time
fdrag = traj.stream.drag_constant * ds79.Fdrag(
    traj.w,
    traj.stream.T,
    ds.phi(traj.X, traj.stream),
    traj.stream.n
)

impulse = np.cumsum(fdrag)*(traj.t[1] - traj.t[0])

# Top graph is same as onaxis_traject
prefix = "figs/on-axis-impulse"
traj.figfile = f'{prefix}-{traj.id_}.pdf'

t0 = traj.t[traj.V >= 0.0].min()
sns.set_style('ticks')
sns.set_color_codes('deep')
fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6))
ax.plot(traj.t - t0, traj.R/traj.Rsd, label=r'$R/R_\mathrm{sd}$')
ax.plot(traj.t - t0, traj.V, label='$v / v_{\infty}$', lw=1.0)

ax.axhspan(0.0, 1.0, color='k', alpha=0.1)
ax.legend(loc="upper right")
ax.set(
    xlabel=r'Time, years',
    ylim=[-1.5, 5.1]
)

# Lower graph shows fdrag to start with
ax2.plot(traj.t - t0, fdrag)
ax2.set(
    xlabel=r'Time, years',
    ylabel=r"Drag force",
)

# Lower graph shows fdrag to start with
ax3.plot(traj.t - t0, impulse)
ax3.set(
    xlabel=r'Time, years',
    ylabel=r"Drag impulse",
)

sns.despine()
fig.tight_layout()
fig.savefig(traj.figfile)
# Prevent resource leaks
plt.close(fig)
print(traj.figfile, end='')

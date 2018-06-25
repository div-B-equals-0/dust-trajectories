import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

dx = dy = 0.01
xmin, xmax = -3.0, 3.0
ymin, ymax = -1.5, 3.5
# xmin, xmax = -2.0, 2.0
# ymin, ymax = -0.2, 1.5
x, y = np.meshgrid(
    np.arange(xmin, xmax, dx),
    np.arange(ymin, ymax, dx)
)
R = np.hypot(x, y)
rho = 1/np.sqrt(1 - 1/R)
rho[y**2 < 1] = 2.0*rho[y**2 < 1]
rho[(y**2 < 1) & (x < 0)] = 0.0
rho[R <= 1] = 0.0

figfile = sys.argv[0].replace(".py", ".png")
sns.set_color_codes()
fig, ax = plt.subplots(figsize=(5, 4))

ax.imshow(rho, origin='lower', extent=[xmin, xmax, ymin, ymax],
          vmin=0.0, vmax=10.0, cmap='gray_r')

ax.axhline(0.0, color="k", alpha=0.5, lw=0.3)
ax.axvline(0.0, color="k", alpha=0.5, lw=0.3)
ax.plot(0.0, 0.0, "*", color="r", ms=6)

ax.set_aspect("equal")
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax],
       xlabel="$x / R_{**}$", ylabel="$y / R_{**}$")
sns.despine()
fig.tight_layout()
fig.savefig(figfile, dpi=600)
print(figfile, end="")

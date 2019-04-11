import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects

import seaborn as sns
import glob
import json

try:
    STARSTRING, VSTRING, DUSTSTRING = sys.argv[1:4]
except:
    print(f"Usage: {sys.argv[0]} STARSTRING VSTRING DUSTSTRING")

pattern = f"data/on-axis-{STARSTRING}-{VSTRING}-n-??-{DUSTSTRING}.json"

# Get all the densities in order
filenames = sorted(glob.glob(pattern))[::-1]
filenames += sorted(glob.glob(pattern.replace("-n-", "-n+")))

density = []
R0 = []
Rstarstar = []
Rsd = []
Red = []
Rmin = []
Rstar = []
# Gas bow shock
Rgbs = []

for fn in filenames:
    data = json.load(open(fn))
    density.append(data["stream"]["n"])
    R0.append(data["stream"]["R0"])
    Rstarstar.append(data["stream"]["Rstarstar"])
    Rsd.append(data["R sonic drift"])
    Red.append(data["R equilib drift"])
    Rmin.append(data["R min"][0])

    Rstar.append(data["stream"]["Rstar"])

    kratio = data["stream"]["kappa"] / data["stream"]["kappa_d"]
    dv_v =  kratio * (data["stream"]["Rstarstar"] /
                      data["R sonic drift"] )
    dv_v *= 0.75
    if dv_v < 1.0:
        Rgbs.append(
            np.sqrt(data["stream"]["eta"])*data["stream"]["Rstar"] / (1.0 - dv_v)
        )
    else:
        Rgbs.append(np.nan)


density = np.array(density)
R0 = np.array(R0)
Rstarstar = np.array(Rstarstar)
Rsd = np.array(Rsd)
Red = np.array(Red)
Rmin = np.array(Rmin)
Rstar = np.array(Rstar)
Rgbs = np.array(Rgbs)


figfile = "figs/" + sys.argv[0].replace(
    ".py", f"-{STARSTRING}-{VSTRING}-{DUSTSTRING}.pdf")
fig, ax = plt.subplots(figsize=(4, 3.5))

m = R0 > Rmin
mlo = density < 1.0
mhi = ~mlo
ax.plot(density[m & mlo], R0[m & mlo], label="Wind bow shock")
m = (1.5*Rmin > R0)
ax.plot(density[m], Rmin[m], label="Dust wave")
m = (1.5*Rmin > R0) & (Rsd < Rstarstar)
ax.fill_between(density[m], Rmin[m], Red[m], color="orange", alpha=0.4)
m = Rgbs <= Rmin
ax.plot(density[m], Rgbs[m], ls="--", label="Dust-free bow shock")
m = R0 > Rmin
mbw = R0 < 0.95*Rstar
ax.plot(density[m & mhi & mbw], R0[m & mhi & mbw], label="Bow wave")

ax.plot(density[m & mhi & ~mbw], R0[m & mhi & ~mbw], label="Radiation bow shock")

m = (3.0*Rsd > Rgbs) & (Rsd < 10.0*Rstarstar)
ax.plot(density[m], Rsd[m], lw=0.3, color="k", label="_nolabel_")
m = density < 10.0
ax.plot(density[m], Rstarstar[m], lw=0.3, color="k", label="_nolabel_")
ax.plot(density, kratio*Rstarstar, lw=0.3, color="k", label="_nolabel_")
# ax.plot(density, Rstar, ls=":", label="Rstar")

box_params = dict(fc='w', ec='0.8', lw=0.4, pad=2)

ax.text(0.001, 1.5, r"$R_{\dag}$",
        ha="left", va="center", bbox=box_params)
ax.text(1e-4, 0.22, r"$R_{**}$",
        ha="left", va="center", bbox=box_params)
ax.text(1e-4, 4e-4, r"$(\kappa / \kappa_\mathrm{d}) \, R_{**} $",
        ha="left", va="center", bbox=box_params)

orange = (0.7, 0.35, 0.0)
ax.text(1.0, 0.1, r"$R_{\ddag}$",
        ha="left", va="center", color=orange,
        path_effects=[PathEffects.withStroke(linewidth=3,
                                             alpha=0.7,
                                             foreground="w")],
        )
ax.text(0.38, 0.04, r"$R_{\mathrm{min}}$",
        ha="left", va="center", color=orange,
        path_effects=[PathEffects.withStroke(linewidth=3,
                                             alpha=0.7,
                                             foreground="w")],
        )

default_title = f"{STARSTRING} {VSTRING} {DUSTSTRING}"
fancy_title = {
    "MS10 v080 gra002":
    r"$M = 10$ M$_\odot$ $v = 80$ km s$^{-1}$ $a = 0.02\ \mu$m graphite ",
}
ax.text(3e-5, 8e-5, fancy_title.get(default_title, default_title),
        ha="left", va="center", fontsize="x-small", bbox=box_params)




leg = ax.legend(loc="upper right", 
                title="Bow variety", fontsize="x-small")
#leg.get_title().set_fontsize("small")

ax.set(
    xscale="log",
    yscale="log",
#    xlim=[1e-4, 2e7],
    xlabel=r"Stream density, $n$, cm$^{-3}$",
    ylabel=r"Bow radius, $R$, parsec",
)
sns.despine()
fig.tight_layout()
fig.savefig(figfile)
print(figfile, end="")

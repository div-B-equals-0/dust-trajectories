import sys
import numpy as np
from matplotlib import pyplot as plt
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
    Rmin.append(data["R min"][0])
    Rstar.append(data["stream"]["Rstar"])

    dv_v =  (data["stream"]["kappa"] /
             data["stream"]["kappa_d"]) * (data["stream"]["Rstarstar"] /
                                           data["R sonic drift"] )
    dv_v *= 0.75
    if dv_v < 1.0:
        Rgbs.append(
            np.sqrt(data["stream"]["eta"])*data["stream"]["Rstar"] / (1.0 - dv_v)
        )
    else:
        Rgbs.append(np.nan)
        
R0 = np.array(R0)
Rstarstar = np.array(Rstarstar)
Rsd = np.array(Rsd)
Rmin = np.array(Rmin)
Rstar = np.array(Rstar)
Rgbs = np.array(Rgbs)

    
figfile = "figs/" + sys.argv[0].replace(
    ".py", f"-{STARSTRING}-{VSTRING}-{DUSTSTRING}.pdf")
fig, ax = plt.subplots()

ax.plot(density, R0/Rstar, label="$R_0 / R_*$")
ax.plot(density, Rgbs/Rstar, label="$R_\mathrm{BS} / R_*$")
ax.plot(density, Rmin/Rstar, label=r"$R_\mathrm{min} / R_*$")
ax.plot(density, Rsd/Rstar, ls="--", label=r"$R_\dag / R_*$")
ax.plot(density, Rstarstar/Rstar, ls=":", label="$R_{**} / R_*$")

# Harmonic mean of Rsd and Rstarstar
# Turned out not to be very accurate 
# Rharm = 1./(1./Rsd + 1./Rstarstar)
# ax.plot(density, Rharm/Rstar, ls=":", label=r"_nolabel_")

ax.legend()
ax.set(
    xscale="log",
    yscale="log",
    xlabel="Stream density, $n$, cm$^{-3}$",
    # ylim=[0.0, 5*R0[0]/Rstar[0]],
    ylim=[None, 1.3],
)
sns.despine()
fig.savefig(figfile)
print(figfile, end="")

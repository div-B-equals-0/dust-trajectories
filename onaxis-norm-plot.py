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

for fn in filenames:
    data = json.load(open(fn))
    density.append(data["stream"]["n"])
    R0.append(data["stream"]["R0"])
    Rstarstar.append(data["stream"]["Rstarstar"])
    Rsd.append(data["R sonic drift"])
    Rmin.append(data["R min"][0])
    Rstar.append(data["stream"]["Rstar"])

R0 = np.array(R0)
Rstarstar = np.array(Rstarstar)
Rsd = np.array(Rsd)
Rmin = np.array(Rmin)
Rstar = np.array(Rstar)

    
figfile = "figs/" + sys.argv[0].replace(
    ".py", f"-{STARSTRING}-{VSTRING}-{DUSTSTRING}.pdf")
fig, ax = plt.subplots()

ax.plot(density, R0/Rstar, label="$R_0 / R_*$")
ax.plot(density, Rmin/Rstar, label=r"$R_\mathrm{DW} / R_*$")
ax.plot(density, Rsd/Rstar, ls="--", label=r"$R_\mathrm{rip} / R_*$")
ax.plot(density, Rstarstar/Rstar, ls=":", label="$R_{**} / R_*$")
ax.legend()
ax.set(
    xscale="log",
    yscale="linear",
    xlabel="Stream density, $n$, cm$^{-3}$",
    ylim=[0.0, 10*R0[0]/Rstar[0]],
)
sns.despine()
fig.savefig(figfile)
print(figfile, end="")

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

filenames = sorted(glob.glob(pattern))[::-1] + sorted(glob.glob(pattern.replace("-n-", "-n+")))

density = []
R0 = []
Rstarstar = []
Rsd = []
Rmin = []

for fn in filenames:
    data = json.load(open(fn))
    density.append(data["stream"]["n"])
    R0.append(data["stream"]["R0"])
    Rstarstar.append(data["stream"]["Rstarstar"])
    Rsd.append(data["R sonic drift"])
    Rmin.append(data["R min"][0])


figfile = "figs/" + sys.argv[0].replace(
    ".py", f"-{STARSTRING}-{VSTRING}-{DUSTSTRING}.pdf")
fig, ax = plt.subplots()

ax.plot(density, R0, label="R0")
ax.plot(density, Rmin, label="Rmin")
ax.plot(density, Rsd, ls="--", label="Rsd")
ax.plot(density, Rstarstar, ls=":", label="R**")
ax.legend()
ax.set(
    xscale="log",
    yscale="log",
#    xlim=[1e-4, 2e7],
)

fig.savefig(figfile)
print(figfile, end="")

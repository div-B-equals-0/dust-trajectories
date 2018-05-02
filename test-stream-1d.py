import sys
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import seaborn as sns
import duststream as ds


try: 
    STAR = sys.argv[1]
    VINF = float(sys.argv[2])
    LOGN = float(sys.argv[3])
    GRAIN = sys.argv[4]
    A = float(sys.argv[5])
except:
    print(f"Usage: {sys.argv[0]} L4 VINF LOGN GRAIN A [ZOOM]")

try: 
    ZOOM = float(sys.argv[6])
except:
    ZOOM = 1.0


L4 = {
    "MS10": 0.64,
    "MS20": 5.45,
    "MS40": 22.2,
    "BSG": 30.2,
}


# Set minor parameters according to star and grain type
Tgas = {
    "MS10": 8000.0,
    "MS20": 9000.0,
    "MS40": 1e4,
    "BSG": 8000.0,
}

if GRAIN == "gra":
    rho_d = 2.2
    if STAR in ["MS10", "BSG"]:
        phi_norm = 1.0
    else:
        phi_norm = 1.5
else:
    rho_d = 3.5
    if STAR in ["MS10", "BSG"]:
        phi_norm = 0.7
    else:
        phi_norm = 1.4




# Initial conditions
Rstart = 2.5/ZOOM
y0 = [Rstart, -1.0]

stream = ds.DustStream(L4=L4[STAR], vinf=VINF, n=10**LOGN, a=A,
                       T=Tgas[STAR], phi_norm=phi_norm, rho_d=rho_d)

streamid = f"{STAR}-v{int(VINF):03d}-n{int(10*LOGN):+02d}-{GRAIN}{int(100*A):03d}"

figfile = sys.argv[0].replace('.py', f'-{streamid}.pdf')

# Time grid
t = np.linspace(0.0, 10.0/ZOOM, 5001)
soln = odeint(ds.dydt_1d, y0, t, args=(stream,))
t0 = t[soln[:, 1] >= 0.0].min()

# Slippage velocity
w = 1.0 + soln[:, 1]
# Drift velocity
# wdrift = 1.0 / alpha / soln[:, 0]

sns.set_style('ticks')
sns.set_color_codes('dark')
fig, (ax, axp) = plt.subplots(2, 1, figsize=(4, 6))
ax.plot(t - t0, soln[:, 0], label='$R/R_{0}$', zorder=3, lw=0.5)
ax.plot(t - t0, soln[:, 1], label='$v / v_{\infty}$', lw=0.5)
#ax.plot(t - t0, wdrift, ls='--', label='$w_\mathrm{drift} / v_{\infty}$')

# ax.axhline(1.0/alpha, ls=':', color='k', lw=0.8)
ax.axhspan(0.0, 1.0, color='k', alpha=0.1)
ax.legend(loc="lower left")
ax.set(
    xlabel=r'Time / $(R_{0} / v_{\infty})$',
    ylim=[-1.1, 2.1]
)
t2 = np.linspace(0.0, 20.0, 201)

x1, x2 = 0.008, 5.0
w1, w2 = 0.03/stream.vinf, 500.0/stream.vinf

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
for z, cmap, dex in [np.log10(agrid), "Blues", 4.0], [np.log10(-agrid), "Reds", 3.0]: 
    axp.contourf(xpts, wpts, z,
                 10, #[-0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
                 vmax=np.nanmax(z), vmin=np.nanmax(z)-dex,
                 cmap=cmap)

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

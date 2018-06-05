import sys
import numpy as np
from onaxis_traject import Trajectory

try: 
    STAR = sys.argv[1]
    GRAIN = sys.argv[2]
    A = float(sys.argv[3])
except:
    sys.exit(f"Usage: {sys.argv[0]} STAR GRAIN A")


logns = np.arange(-4.3, 7.0, 0.1)


vs = [80.0,]

for logn in logns:
    print(f'Log n = {logn:.1f}')
    for vinf in vs:
        traj = Trajectory(STAR, vinf, logn, GRAIN, A)
        traj.integrate()
        traj.savedata("data/on-axis")
#        traj.savefig("figs/on-axis")
        

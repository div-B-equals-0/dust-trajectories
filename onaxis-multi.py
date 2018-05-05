import sys
import numpy as np
from onaxis_traject import Trajectory

try: 
    STAR = sys.argv[1]
    GRAIN = sys.argv[2]
    A = float(sys.argv[3])
except:
    print(f"Usage: {sys.argv[0]} STAR GRAIN A")



logns = np.arange(-3.0, 5.1, 0.1)
vs = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 140.0, 170.0, 200.0, 250.0, 300.0, 500.0]

for logn in logns:
    print(f'Log n = {logn:.1f}')
    for vinf in vs:
        traj = Trajectory(STAR, vinf, logn, GRAIN, A)
        traj.integrate()
        traj.savedata("data/on-axis")
#        traj.savefig("figs/on-axis")
        

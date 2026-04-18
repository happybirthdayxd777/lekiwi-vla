#!/usr/bin/env python3
import h5py, numpy as np

files = [
    'data/jacobian_pctrl_50ep_kP01.h5',
    'data/jacobian_pctrl_100ep_kP01.h5', 
    'data/jacobian_pctrl_50ep_kP01_v2.h5',
]
all_goals = []
for f in files:
    try:
        with h5py.File(f, 'r') as hf:
            goals = hf['goal_positions'][:]
            all_goals.append(goals)
            n = len(goals)
            gx, gy = goals[:,0], goals[:,1]
            q = {
                '+X+Y': int(((gx>=0)&(gy>=0)).sum()),
                '+X-Y': int(((gx>=0)&(gy<0)).sum()),
                '-X+Y': int(((gx<0)&(gy>=0)).sum()),
                '-X-Y': int(((gx<0)&(gy<0)).sum()),
            }
            print(f'{f}: {n}f, +X+Y={q["+X+Y"]}({q["+X+Y"]/n*100:.0f}%), +X-Y={q["+X-Y"]}({q["+X-Y"]/n*100:.0f}%), -X+Y={q["-X+Y"]}({q["-X+Y"]/n*100:.0f}%), -X-Y={q["-X-Y"]}({q["-X-Y"]/n*100:.0f}%)')
    except Exception as e:
        print(f'{f}: ERROR {e}')

print()
all_g = np.vstack(all_goals)
gx, gy = all_g[:,0], all_g[:,1]
n = len(all_g)
q = {
    '+X+Y': int(((gx>=0)&(gy>=0)).sum()),
    '+X-Y': int(((gx>=0)&(gy<0)).sum()),
    '-X+Y': int(((gx<0)&(gy>=0)).sum()),
    '-X-Y': int(((gx<0)&(gy<0)).sum()),
}
print(f'COMBINED: {n}f, +X+Y={q["+X+Y"]}({q["+X+Y"]/n*100:.0f}%), +X-Y={q["+X-Y"]}({q["+X-Y"]/n*100:.0f}%), -X+Y={q["-X+Y"]}({q["+X+Y"]/n*100:.0f}%), -X-Y={q["-X-Y"]}({q["-X-Y"]/n*100:.0f}%)')

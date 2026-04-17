#!/usr/bin/env python3
"""Diagnose why P-controller fails on LeKiWiSim (primitive) vs succeeds on URDF sim."""
import sys, os, numpy as np
sys.path.insert(0, '.'); os.chdir('.')

from sim_lekiwi_urdf import twist_to_contact_wheel_speeds
from sim_lekiwi_urdf import LeKiWiSimURDF
from sim_lekiwi import LeKiwiSim

goal = np.array([0.5, 0.0])
class P:
    def __init__(self,kP=1.5,max_speed=0.3,wheel_clip=0.5):
        self.kP=kP;self.max_speed=max_speed;self.wheel_clip=wheel_clip
    def compute(self,base_pos,goal_pos):
        dx,dy=goal_pos[0]-base_pos[0],goal_pos[1]-base_pos[1]
        dist=np.linalg.norm([dx,dy])
        if dist<0.05: return np.zeros(3)
        v_mag=min(self.kP*dist,self.max_speed)
        ws=twist_to_contact_wheel_speeds(v_mag*dx/dist,v_mag*dy/dist)
        return np.clip(ws,-self.wheel_clip,self.wheel_clip)

print("="*60)
print("URDF SIM TEST")
print("="*60)
sim=LeKiWiSimURDF(); p=P()
for ep in range(3):
    sim.reset()
    for step in range(200):
        bp=sim.data.qpos[:2].copy()
        action=np.zeros(9,dtype=np.float32); action[6:9]=p.compute(bp,goal)
        sim.step(action)
        d=np.linalg.norm(goal-sim.data.qpos[:2])
        if d<0.3: print(f'URDF ep{ep}: SUCCESS step={step}'); break
    else: print(f'URDF ep{ep}: FAIL final_dist={d:.3f}')

print()
print("="*60)
print("PRIMITIVE SIM TEST")
print("="*60)
sim2=LeKiwiSim(); p2=P()
for ep in range(3):
    sim2.reset()
    for step in range(200):
        bp=sim2.data.qpos[:2].copy()
        action=np.zeros(9,dtype=np.float32); action[6:9]=p2.compute(bp,goal)
        sim2.step(action)
        d=np.linalg.norm(goal-sim2.data.qpos[:2])
        if d<0.3: print(f'Prim ep{ep}: SUCCESS step={step}'); break
    else: print(f'Prim ep{ep}: FAIL final_dist={d:.3f}, final_pos={sim2.data.qpos[:2]}')

print()
print("="*60)
print("DRIVE FORCE COMPARISON")
print("="*60)
# Test: apply same action to both, measure resulting world velocity
for name,sim in [('URDF',LeKiWiSimURDF()),('Prim',LeKiwiSim())]:
    sim.reset()
    # Apply max wheel action for 50 steps, measure displacement
    for _ in range(50):
        action=np.zeros(9,dtype=np.float32); action[6:9]=[0.5,0.5,0.5]
        sim.step(action)
    disp=np.linalg.norm(sim.data.qpos[:2])
    print(f'{name}: after 50 steps of [0.5,0.5,0.5], disp={disp:.4f}m')
    sim.reset()
    # Apply twist_to_contact wheel speeds for typical drive
    ws=twist_to_contact_wheel_speeds(0.3,0.0)
    for _ in range(50):
        action=np.zeros(9,dtype=np.float32); action[6:9]=np.clip(ws,-0.5,0.5)
        sim.step(action)
    disp2=np.linalg.norm(sim.data.qpos[:2])
    print(f'{name}: after 50 steps of twist(0.3,0), disp={disp2:.4f}m')
    print(f'{name}: qpos[:3]={sim.data.qpos[:3]}')
    print()

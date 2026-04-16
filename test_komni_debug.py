#!/usr/bin/env python3
"""Debug k_omni physics in LeKiWiSimURDF."""
import numpy as np
from sim_lekiwi_urdf import LeKiWiSimURDF, _omni_kinematics

sim = LeKiWiSimURDF()
print(f"nq={sim.model.nq}, nv={sim.model.nv}, nctrl={sim.model.nctrl}")

# Check joints
jnt_names = [sim.model.joint_id2name(i) for i in range(sim.model.njnt)]
print(f"joints: {jnt_names}")

# Check dofadr for wheels
for name in ['w1', 'w2', 'w3']:
    jid = sim.model.joint(name).id
    adr = sim.model.joint(name).dofadr[0]
    print(f"{name}: joint_id={jid}, dofadr={adr}")

# Reset and apply X-drive action (wheel1=+0.5, wheel2=-0.5, wheel3=0)
sim.reset()
base_id = sim.model.body('base').id

print("\n--- Testing X-drive action [0.5, -0.5, 0.0] ---")
for step in range(10):
    obs, r, done, info = sim.step(np.array([0.5, -0.5, 0.0, 0, 0, 0, 0, 0, 0]))
    
    # Get wheel velocities
    w1_vel = sim.data.qvel[sim.model.joint('w1').dofadr[0]]
    w2_vel = sim.data.qvel[sim.model.joint('w2').dofadr[0]]
    w3_vel = sim.data.qvel[sim.model.joint('w3').dofadr[0]]
    
    # Get base position
    base_pos = sim.data.xpos[base_id]
    
    # Get ctrl
    ctrl = sim.data.ctrl
    print(f"step {step}: base=({base_pos[0]:.4f},{base_pos[1]:.4f},{base_pos[2]:.4f}), "
          f"w_vels=[{w1_vel:.3f},{w2_vel:.3f},{w3_vel:.3f}], "
          f"ctrl_w=[{ctrl[6]:.2f},{ctrl[7]:.2f},{ctrl[8]:.2f}]")

print("\n--- Direct _omni_kinematics test ---")
# What wheel velocities give good omni-kinematics?
test_wheel_vels = [[5.0, -5.0, 0.0], [2.5, 2.5, -2.5], [1.0, 1.0, -1.0]]
for wv in test_wheel_vels:
    vx, vy, wz = _omni_kinematics(np.array(wv))
    print(f"wheels={wv}: vx={vx:.4f}, vy={vy:.4f}, wz={wz:.4f}")
    print(f"  k_omni=15: Fx={15*vx:.3f}N, Fy={15*vy:.3f}N")

print("\n--- Check current step output ---")
# Why does step give near-zero wheel velocities?
# Let's check the action_to_ctrl conversion
action = np.array([0.5, -0.5, 0.0, 0, 0, 0, 0, 0, 0])
ctrl = sim._action_to_ctrl(action)
print(f"action[6:9]={action[6:9]}")
print(f"ctrl[6:9]={ctrl[6:9]} (wheel torques)")
print(f"With gear=10: joint torques = {ctrl[6:9]*10}")

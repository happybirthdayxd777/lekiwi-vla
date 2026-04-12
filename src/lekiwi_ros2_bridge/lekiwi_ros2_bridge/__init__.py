# LeKiWi ROS2 Bridge — Python Package
# Bridges ROS2 topics <-> MuJoCo simulation for LeKiWi VLA research platform
#
# Nodes:
#   bridge_node.py     — ROS2<->MuJoCo bridge (cmd_vel, joint_states, cameras)
#   vla_policy_node.py — CLIP-FM policy inference over ROS2 topics
#   replay_node.py     — HDF5 trajectory replay over ROS2
#   security_monitor.py — CTF intrusion detection (speed spikes, HMAC, replay)
#   policy_guardian.py — Active policy defense (whitelist, rollback, alerts)
#
# Utilities:
#   lekiwi_sim_loader.py — Factory: LeKiWiSim / LeKiWiSimURDF / RealHardwareAdapter
#   real_hardware_adapter.py — Serial bus adapter for real ST3215 servos

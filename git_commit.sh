#!/bin/bash
set -e
cd ~/hermes_research/lekiwi_vla
git add -A
git commit -m "ROS2 bridge auto heartbeat $(date '+%Y%m%d %H%M') — Phase 1-6 complete"
git push origin main

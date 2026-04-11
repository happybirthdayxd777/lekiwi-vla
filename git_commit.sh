#!/bin/bash
set -e
cd ~/lekiwi_vla
git add -A
git commit -m "Add Gymnasium env wrapper, LeRobot policy inference (fixed Python 3.13 compat), mock policy demo"
git push origin main
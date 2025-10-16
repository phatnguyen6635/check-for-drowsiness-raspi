#!/bin/bash
set -e

# Wait a bit to make sure the system is ready
sleep 5

# ----------------------------
# Load Conda environment
# ----------------------------

source /home/raspi/miniconda3/etc/profile.d/conda.sh
conda activate check-for-drowsiness

# ----------------------------
# Run main Python script
# ----------------------------
cd /home/raspi/Documents/project/check-for-drowsiness-raspi || exit
python tools/main.py

#!/bin/bash
set -e

# Wait a bit to make sure the system is ready
sleep 5

# ----------------------------
# Load Conda environment
# ----------------------------

source /home/jetson/miniconda3/etc/profile.d/conda.sh
conda activate check-for-drowsiness

# ----------------------------
# Run main Python script
# ----------------------------
cd /home/jetson/Documents/repo/check-for-drowsiness || exit
python tools/main.py >> /home/jetson/start.log 2>&1

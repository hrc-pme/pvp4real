#!/bin/bash
set -e

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Source local workspace if available
if [ -f "/workspace/colcon-ws/install/setup.bash" ]; then
    source /workspace/colcon-ws/install/setup.bash
fi

echo '=========================================='
echo 'Starting PVP HITL (Human-in-the-Loop)...'
echo '=========================================='

# Run the HITL script
cd /workspace
python3 pvp4real/scripts/pvp.hitl.py

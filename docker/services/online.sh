#!/bin/bash
set -e

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Source local workspace if available
if [ -f "/workspace/colcon-ws/install/setup.bash" ]; then
    source /workspace/colcon-ws/install/setup.bash
fi

echo '=========================================='
echo 'Starting PVP Online Training (HITL)...'
echo '=========================================='

cd /workspace

# Launch stretch3 authority arbitration node in background
echo '[online.sh] Starting stretch3.hitl.py (authority arbitration)...'
python3 pvp4real/scripts/train/online/stretch3.hitl.py &
STRETCH3_PID=$!

# Give stretch3 node time to initialise before PVP connects
sleep 2

# Run PVP training in foreground; wait for it to finish
echo '[online.sh] Starting pvp.hitl.py (HITL training)...'
python3 pvp4real/scripts/train/online/pvp.hitl.py
PVP_EXIT=$?

# Clean up stretch3 background process
echo '[online.sh] pvp.hitl.py exited (code: '"$PVP_EXIT"'). Stopping stretch3.hitl.py...'
kill "$STRETCH3_PID" 2>/dev/null && wait "$STRETCH3_PID" 2>/dev/null || true

exit "$PVP_EXIT"

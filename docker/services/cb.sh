#!/bin/bash
set -e

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Start colcon build
echo 'Starting colcon build...'
colcon build

# Success message
echo 'Build completed successfully!'
exit 0
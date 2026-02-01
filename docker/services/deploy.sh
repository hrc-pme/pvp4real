#!/bin/bash
set -e

# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Deploy service entry point
echo 'Starting deployment service...'

# Keep container running
exec bash

#!/bin/bash
# Cleanup script to kill lingering ROS2/Gazebo processes

echo "Cleaning up ROS2 and Gazebo processes..."

# Kill gazebo processes
killall -9 gz 2>/dev/null
killall -9 gzserver 2>/dev/null
killall -9 gzclient 2>/dev/null
killall -9 sim 2>/dev/null

# Kill ROS2 bridge processes
killall -9 parameter_bridge 2>/dev/null
killall -9 create 2>/dev/null

# Kill ROS2 nodes
killall -9 sync_slam_toolbox_node 2>/dev/null
killall -9 robot_state_publisher 2>/dev/null
killall -9 scan_fix 2>/dev/null
killall -9 gt_pose_publisher 2>/dev/null
killall -9 odom_corruptor 2>/dev/null
killall -9 auto_driver 2>/dev/null
killall -9 lifecycle_manager 2>/dev/null
killall -9 eval_node 2>/dev/null

# Wait for processes to fully terminate
sleep 2

# Clear ROS2 runtime directories if needed
rm -rf ~/.ros/run/* 2>/dev/null

echo "Cleanup complete!"

#!/bin/bash
# Cleanup script to kill lingering ROS2/Gazebo processes

echo "Cleaning up ROS2 and Gazebo processes..."

# Kill Gazebo Harmonic processes (gz sim)
pkill -9 -f "gz sim" 2>/dev/null
pkill -9 -f "gz-sim" 2>/dev/null
pkill -9 -f "ruby.*gz" 2>/dev/null
killall -9 gz 2>/dev/null
killall -9 gzserver 2>/dev/null
killall -9 gzclient 2>/dev/null
killall -9 ruby 2>/dev/null  # gz often runs via ruby wrapper

# Kill specific Gazebo Harmonic server/gui
pkill -9 -f "gz-sim-server" 2>/dev/null
pkill -9 -f "gz-sim-gui" 2>/dev/null
pkill -9 -f "gz-gui" 2>/dev/null
pkill -9 -f "ign gazebo" 2>/dev/null
pkill -9 -f "ignition" 2>/dev/null

# Kill ROS2 bridge processes
killall -9 parameter_bridge 2>/dev/null
pkill -9 -f "ros_gz_bridge" 2>/dev/null
pkill -9 -f "ros_gz_sim" 2>/dev/null
killall -9 create 2>/dev/null

# Kill ROS2 nodes from our packages
killall -9 sync_slam_toolbox_node 2>/dev/null
killall -9 robot_state_publisher 2>/dev/null
killall -9 scan_fix 2>/dev/null
killall -9 gt_pose_publisher 2>/dev/null
killall -9 odom_corruptor 2>/dev/null
killall -9 auto_driver 2>/dev/null
killall -9 lifecycle_manager 2>/dev/null
killall -9 eval_node 2>/dev/null
killall -9 dataset_recorder 2>/dev/null
killall -9 train_model 2>/dev/null
killall -9 infer_node 2>/dev/null

# Kill any remaining Python ROS2 nodes
pkill -9 -f "ros2" 2>/dev/null
pkill -9 -f "_ros2_daemon" 2>/dev/null

# Wait for processes to fully terminate
sleep 1

# Force kill any remaining gazebo-related processes
pgrep -f "gz|gazebo|ignition" | xargs -r kill -9 2>/dev/null

# Wait a bit more
sleep 1

# Clear ROS2 runtime directories if needed
rm -rf ~/.ros/run/* 2>/dev/null

# Clear Gazebo cache that might cause issues
rm -rf ~/.gz/fuel/fuel.gazebosim.org/cache/* 2>/dev/null

# NIE USUWAJ poprzednich eksperymentów - są świętością!
# Poprzednie eksperymenty są zapisane w podfolderach exp_YYYYMMDD_HHMMSS
# i nie powinny być modyfikowane ani usuwane

# Tylko wyczyść pliki tymczasowe z głównego katalogu out (nie z podfolderów exp_*)
OUT_DIR="$HOME/SLAM_AI/ai_slam_ws/out"
if [ -d "$OUT_DIR" ]; then
    echo "Czyszczenie plików tymczasowych z $OUT_DIR (bez usuwania eksperymentów)..."
    # Usuń tylko pliki bezpośrednio w out/, nie w podfolderach
    rm -f "$OUT_DIR/model.pt" "$OUT_DIR/dataset.npz" "$OUT_DIR/train_history.json" 2>/dev/null
    rm -f "$OUT_DIR/results.json" "$OUT_DIR/experiment_metadata.json" 2>/dev/null
    rm -f "$OUT_DIR"/*.png 2>/dev/null
    # NIE usuwaj exp_* - to są zapisane eksperymenty!
    # rm -rf "$OUT_DIR"/exp_* 2>/dev/null  # ZAKOMENTOWANE - nie usuwaj!
fi

echo "Cleanup complete! Poprzednie eksperymenty zachowane."

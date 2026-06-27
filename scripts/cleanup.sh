#!/bin/bash
# Cleanup script to kill lingering ROS2/Gazebo processes

echo "Cleaning up ROS2 and Gazebo processes..."

# Kill Gazebo Harmonic processes (gz sim)
# Szeroki sweep "gz" (jak wcześniej), ale z filtrem PID:
# nie ubijamy samego cleanup.sh ani procesu rodzica.
pgrep -f "gz" \
  | awk -v self="$$" -v parent="$PPID" '$1 != self && $1 != parent { print $1 }' \
  | xargs -r kill -9 2>/dev/null || true

pkill -9 -f "gz sim" 2>/dev/null
pkill -9 -f "gz-sim" 2>/dev/null
pkill -9 -f "ruby.*gz" 2>/dev/null
killall -9 gz 2>/dev/null
killall -9 gzserver 2>/dev/null
killall -9 gzclient 2>/dev/null
# Nie ubijamy globalnie wszystkich procesów ruby (zbyt agresywne).
# Gazebo-ruby zamykamy selektywnie przez wzorzec "ruby.*gz" powyżej.

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
killall -9 planned_path_driver 2>/dev/null
killall -9 dataset_motion_watchdog 2>/dev/null
killall -9 lifecycle_manager 2>/dev/null
killall -9 eval_node 2>/dev/null
killall -9 dataset_recorder 2>/dev/null
killall -9 train_model 2>/dev/null
killall -9 infer_node 2>/dev/null
# Instalowane executables mają osobne nazwy (train_model_robak itd.) — killall train_model ich nie łapie
pkill -9 -f "/ai_slam_ai/train_model" 2>/dev/null
pkill -9 -f "/ai_slam_ai/dataset_recorder" 2>/dev/null
pkill -9 -f "/ai_slam_ai/infer_" 2>/dev/null
pkill -9 -f "/ai_slam_bringup/scan_matcher" 2>/dev/null
pkill -9 -f "/ai_slam_bringup/planned_path_driver" 2>/dev/null
pkill -9 -f "/ai_slam_bringup/dataset_motion_watchdog" 2>/dev/null

# Kill any remaining Python ROS2 nodes
pkill -9 -f "ros2" 2>/dev/null
pkill -9 -f "_ros2_daemon" 2>/dev/null

# Wait for processes to fully terminate
sleep 1

# Force kill any remaining Gazebo-related processes.
pgrep -f "(gz|gazebo|ign gazebo|ignition)" \
  | awk -v self="$$" -v parent="$PPID" '$1 != self && $1 != parent { print $1 }' \
  | xargs -r kill -9 2>/dev/null || true

# Wait a bit more
sleep 1

# Final hard check: nie zostawiaj osieroconych procesów środowiska.
for _attempt in 1 2 3; do
  _stale_pids="$(
    pgrep -f "(gz|gazebo|ign gazebo|ignition|ros2 launch ai_slam_bringup demo.launch.py|/ros_gz_bridge/parameter_bridge|/ai_slam_ai/dataset_recorder|/ai_slam_bringup/planned_path_driver|/ai_slam_bringup/dataset_motion_watchdog|/ai_slam_bringup/odom_corruptor|/ai_slam_bringup/gt_pose_publisher|/ai_slam_bringup/scan_fix)" \
      | awk -v self="$$" -v parent="$PPID" '$1 != self && $1 != parent { print $1 }' \
      || true
  )"
  if [ -z "${_stale_pids}" ]; then
    break
  fi
  echo "Cleanup retry ${_attempt}: usuwam osierocone PID: ${_stale_pids}" >&2
  echo "${_stale_pids}" | xargs -r kill -9 2>/dev/null || true
  sleep 0.5
done

# Stop ROS2 daemon cleanly (if running)
ros2 daemon stop >/dev/null 2>&1 || true

# Clear stale FastDDS SHM lock files AFTER all kill-9 (kolejność krytyczna dla WSL2).
# Czyszczenie PRZED kill-9 jest nieskuteczne — procesy mogą odtworzyć pliki SHM
# między czyszczeniem a śmiercią.
find /dev/shm -maxdepth 1 -type f \
  \( -name 'fastrtps_*' -o -name 'fastrtps_port*' -o -name 'sem.fastrtps_port*_mutex' \) \
  -delete 2>/dev/null || true
find /dev/shm -maxdepth 1 -type f -name 'Fast*' -delete 2>/dev/null || true

# Clear ROS2 runtime directories if needed
rm -rf ~/.ros/run/* 2>/dev/null

# Clear Gazebo cache that might cause issues
rm -rf ~/.gz/fuel/fuel.gazebosim.org/cache/* 2>/dev/null

# Dodatkowy sleep na zwolnienie GPU/SHM zasobów (WSL2 z NVIDIA potrzebuje czasu)
sleep 2

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

# AI SLAM (ROS 2 Jazzy + Gazebo Harmonic + slam_toolbox + PyTorch)

Repo zawiera kompletny workspace `ai_slam_ws/` z symulacją robota różnicowego z LIDAR 2D (360 próbek/obrót), baseline SLAM (slam_toolbox) oraz modułem AI do korekcji dryfu, wraz z ewaluacją (RMSE trajektorii i IoU mapy).

## Wymagania
- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Harmonic (gz-sim)
- GPU nie jest wymagana (PyTorch CPU)

## Instalacja zależności
```bash
chmod +x ./scripts/install_deps.sh
./scripts/install_deps.sh
```
## Build
```bash
source /opt/ros/jazzy/setup.bash
source ./.venv/bin/activate

cd ai_slam_ws
rosdep install --from-paths src --ignore-src -r -y --skip-keys ament_python
colcon build --symlink-install
source install/setup.bash
```

## Uruchomienie baseline (Gazebo + robot + bridge + SLAM + automatyczny driver + ewaluacja)
```bash
cd ai_slam_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch ai_slam_bringup demo.launch.py mode:=baseline seed:=123 duration_sec:=120
```
## Wyniki:
- ./out/results.json
- ./out/trajectory.png
- ./out/errors.png
- ./out/maps.png
- ./out/dataset.npz (zapis danych do uczenia w trakcie jazdy)

## Uruchomienie baseline + AI (auto-trening jeśli brak modelu)
```bash
cd ai_slam_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch ai_slam_bringup demo.launch.py mode:=ai seed:=123 duration_sec:=180 dataset_duration_sec:=45
```
## Wyniki:
- ./out/model.pt
- ./out/train_history.json
- ./out/results.json
- wykresy w ./out/

## Ręczne uruchomienie modułu AI
1. Zbieranie datasetu:
```bash
cd ai_slam_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run ai_slam_ai dataset_recorder --ros-args -p out_dir:=out -p duration_sec:=60 -p seed:=123
```
2. Trening:
```bash
ros2 run ai_slam_ai train_model --ros-args -p out_dir:=out -p seed:=123
```
3. Inferencja online:
```bash
ros2 run ai_slam_ai infer_node --ros-args -p out_dir:=out -p seed:=123
```
## Tematy (ROS)
- /scan (z bridge, LaserScan 360)
- /scan_slam (po korekcji formatu)
- /odom_raw (odometria z symulacji, traktowana jako ground truth)
- /odom (odometria z deterministycznym dryfem/szumem)
- /ground_truth_pose (PoseStamped z /odom_raw)
- /cmd_vel
- /pose_ai (PoseStamped korekcji AI)
- /map (slam_toolbox baseline)
- /map_ai (slam_toolbox z odometrią skorygowaną AI)
- /clock
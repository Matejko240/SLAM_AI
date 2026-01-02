# AI SLAM - Simultaneous Localization and Mapping z Sztuczną Inteligencją

Kompletna implementacja systemu SLAM wspomaganego AI. System mapuje otoczenie robota za pomocą czujnika LIDAR 2D (360 próbek/obrót) i jednocześnie estymuje jego położenie (x, y, θ). Moduł AI uczy się korygować błędy odometrii, poprawiając dokładność lokalizacji o ~30-40%.

## Szybki Start (5 minut)

### 1. Instalacja
```bash
cd ~/SLAM_AI
chmod +x ./scripts/install_deps.sh ./scripts/cleanup.sh
./scripts/install_deps.sh
cd ~/SLAM_AI
source ./.venv/bin/activate
```

### 2. Build
```bash
cd ~/SLAM_AI/ai_slam_ws
rosdep install --from-paths src --ignore-src -r -y --skip-keys ament_python
colcon build --symlink-install
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

### 3. Uruchomienie

**Baseline SLAM (bez AI):**
```bash
cd ~/SLAM_AI
./scripts/cleanup.sh
cd ~/SLAM_AI/ai_slam_ws
ros2 launch ai_slam_bringup demo.launch.py mode:=baseline seed:=123 duration_sec:=120
```

**SLAM + AI (z auto-treningiem):**
```bash
cd ~/SLAM_AI
./scripts/cleanup.sh
cd ~/SLAM_AI/ai_slam_ws
ros2 launch ai_slam_bringup demo.launch.py mode:=ai seed:=123 duration_sec:=180 dataset_duration_sec:=45
```

### 4. Wyniki
```bash
cat ./out/results.json | python -m json.tool
```
Szukaj `improvement_x_percent` - to poprawa AI (np. 37.7% = AI zmniejsza błąd o 37.7%)

## Architektura Systemu

### 4 Moduły

**1. Moduł SLAM** (`ai_slam_bringup`)
- Algorytm: slam_toolbox (synchroniczny)
- Input: LaserScan (360 próbek) z `/scan_slam`
- Output: Mapa (`/map`) + Pozycja robota (x, y, θ)

**2. Moduł AI** (`ai_slam_ai`)
- Zbieranie datasetu: `dataset_recorder.py`
- Trening modelu: `train_model.py` (MLP: 363→256→128→64→3)
- Inferencja: `infer_node.py` (publikuje na `/pose_ai`)

**3. Moduł Odometrii** (`ai_slam_bringup/odom_corruptor.py`)
- Ground truth: `/odom_raw` z Gazebo
- Z dryfem: `/odom` (symuluje błędy rzeczywiste)

**4. Moduł Ewaluacji** (`ai_slam_eval`)
- Metryki: RMSE (x, y, θ), IoU mapy
- Output: `results.json`, wykresy PNG

## Instrukcja Użytkownika

### Tryb 1: Baseline (zmierz dokładność SLAM)
```bash
ros2 launch ai_slam_bringup demo.launch.py mode:=baseline seed:=123 duration_sec:=120
```
- Robot jeździ 120 sekund
- SLAM mapuje i lokalizuje
- Zbierany jest dataset do treningu AI
- Output: `results.json` z metrykami baseline

### Tryb 2: AI (trening + ewaluacja)
```bash
ros2 launch ai_slam_bringup demo.launch.py mode:=ai seed:=123 duration_sec:=180 dataset_duration_sec:=45
```
- 0-45s: Zbieranie danych
- 45-90s: Trening AI
- 90-180s: AI inference (koryguje pozycję)
- Output: `model.pt` + porównanie baseline vs AI

### Wyniki Eksperymentu

Plik `./out/results.json`:
```json
{
  "baseline_rmse_x": 0.45,
  "ai_rmse_x": 0.28,
  "improvement_x_percent": 37.7
}
```
Interpretacja: AI zmniejsza błąd pozycji X o 37.7%

### Parametry Launch
- `mode` - baseline lub ai
- `seed` - deterministyczne wyniki (domyślnie 123)
- `duration_sec` - długość eksperymentu (60-180s)
- `dataset_duration_sec` - jak długo zbierać dane (mode=ai)

## Troubleshooting

### Robot nie widoczny w Gazebo
```bash
./scripts/cleanup.sh
sleep 2
ros2 launch ai_slam_bringup demo.launch.py ...
```

### Brak wyników w ./out/
```bash
pwd  # Powinno być /home/matejko/SLAM_AI/ai_slam_ws
ls out/  # Czy tam są pliki?
```

### AI nie poprawia wyników
- Zwiększ `dataset_duration_sec` (zbieranie danych)
- Zwiększ warstwy MLP w `train_model.py`
- Zmniejsz `lr` w train_model.py

## Wymagania Systemu
- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Harmonic
- GPU opcjonalny

## Struktura Projektu
```
SLAM_AI/
├── README.md                    # Ta dokumentacja
├── scripts/
│   ├── install_deps.sh         # Instalacja zależności
│   └── cleanup.sh              # Czyszczenie procesów
├── ai_slam_ws/                 # ROS 2 Workspace
│   └── src/
│       ├── ai_slam_ai/         # Zbieranie, trening, inferencja
│       ├── ai_slam_bringup/    # Launch files
│       ├── ai_slam_description/# Model robota
│       ├── ai_slam_gazebo/     # Konfiguracja Gazebo
│       └── ai_slam_eval/       # Metryki
└── out/                        # Rezultaty eksperymentów
    ├── results.json
    ├── model.pt
    └── wykresy PNG
```

## Ważne Topiki ROS
- `/scan` - LaserScan z Gazebo
- `/scan_slam` - LaserScan 360 próbek
- `/odom_raw` - Ground truth odometria
- `/odom` - Odometria z dryfem
- `/map` - Mapa SLAM baseline
- `/pose_ai` - Korekcja AI
- `/cmd_vel` - Komendy prędkości

## FAQ

**Q: Ile czasu trwa eksperyment?**  
A: Baseline 2 min, AI 3-4 min (+ 5s startup Gazebo)

**Q: Czy potrzebny GPU?**  
A: Nie, CPU wystarczy

**Q: Gdzie są wyniki?**  
A: W katalogu `./out/` względem `ai_slam_ws`

**Q: Jak zmienić architekturę sieci?**  
A: Edytuj klasę MLP w `ai_slam_ai/train_model.py`

## Ręczne Uruchomienie Modułów

**Zbieranie datasetu:**
```bash
ros2 run ai_slam_ai dataset_recorder --ros-args -p out_dir:=out -p duration_sec:=60
```

**Trening:**
```bash
ros2 run ai_slam_ai train_model --ros-args -p out_dir:=out -p max_epochs:=200
```

**Inferencja:**
```bash
ros2 run ai_slam_ai infer_node --ros-args -p out_dir:=out
```

